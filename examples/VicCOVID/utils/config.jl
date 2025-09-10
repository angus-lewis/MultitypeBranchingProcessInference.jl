include("../../utils/gaussianprocesses.jl")
include("../../utils/loglikelihoodutils.jl")
include("../../utils/generalutils.jl")
include("../../utils/mhutils.jl")

function makemodel(config)
    seirconfig = config["model"]["stateprocess"]["params"]
    delta, lambda, beta = convertseirparamstorates(
        first(seirconfig["R_0"]), first(seirconfig["T_E"]), first(seirconfig["T_I"]),
        seirconfig["E_state_count"], seirconfig["I_state_count"],
    )
    seir = SEIR(
        seirconfig["E_state_count"], 
        seirconfig["I_state_count"],
        beta, 
        delta, 
        lambda,
        seirconfig["observation_probability"], 
        nothing, # notification rate
        seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"],
        config["model"]["stateprocess"]["initial_state"],
    )

    obs_config = config["model"]["observation"]
    obs_operator = zeros(1, getntypes(seir))
    obs_state_idx = seirconfig["E_state_count"] + seirconfig["I_state_count"] + 1
    obs_operator[obs_state_idx] = 1.0
    obs_model = LinearGaussianObservationModel(
        obs_operator, obs_config["mean"], 
        reshape(obs_config["cov"], 1, 1)
    )

    model = StateSpaceModel(seir, obs_model)

    param_vec = MTBPParams{paramtype(model), Vector{paramtype(model)}}[]
    param_seq = MTBPParamsSequence(param_vec)
    if seirconfig["is_time_homogeneous"]
        mtbpparams = MTBPParams(seir)
        R_0 = seirconfig["R_0"]
        T_E = seirconfig["T_E"]
        T_I = seirconfig["T_I"]
        immigration = seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"]
        param_map!(
            mtbpparams, 
            seirconfig["E_state_count"], 
            seirconfig["I_state_count"], 
            (R_0, T_E, T_I), 
            immigration, 
        )
        push!(param_seq.seq, mtbpparams)
    else
        for i in eachindex(seirconfig["R_0"])
            mtbpparams = MTBPParams(seir)
            paramtimestamp = seirconfig["timestamps"][i]
            R_0 = seirconfig["R_0"][i]
            T_E = seirconfig["T_E"][i]
            T_I = seirconfig["T_I"][i]
            immigration = seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"]
            param_map!(
                mtbpparams, 
                seirconfig["E_state_count"], 
                seirconfig["I_state_count"], 
                (R_0, T_E, T_I),
                immigration, 
            )
            mtbpparams.time = paramtimestamp
            push!(param_seq.seq, mtbpparams)
        end
    end
    return model, param_seq
end

function makeloglikelihood(observations, config)
    model, param_seq = makemodel(config)
    if config["inference"]["likelihood_approx"]["method"] == "kalman_filter"
        approx = MTBPKalmanFilterApproximation(model)
        if "switch" in keys(config["inference"]["likelihood_approx"])
            @warn "Unused \"switch\" params in config with approximation method \"kalman_filter\"."
        end
        if "particle_filter" in keys(config["inference"]["likelihood_approx"])
            @warn "Unused \"particle_filter\" params in config with approximation method \"kalman_filter\"."
        end
    else
        error("Unknown likelihood_approx method specified in config.")
    end

    seirconfig = config["model"]["stateprocess"]["params"]
    llparam_map! = (mtbpparams, R0, i) -> begin
        return param_map!(
            mtbpparams, 
            seirconfig["E_state_count"], 
            seirconfig["I_state_count"], 
            (R0, seirconfig["T_E"][i], seirconfig["T_I"][i]), 
            seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"], 
        )
    end

    reset_idx = seirconfig["E_state_count"] + seirconfig["I_state_count"] + 1
    reset_obs_state_iter_setup! = (f, model, dt, observation, iteration, use_prev_iter_params) -> begin
        reset_state_iter_setup!(f, model, dt, observation, iteration, use_prev_iter_params, reset_idx)
        return
    end

    loglikelihood = (pars) -> begin # function loglikelihood(pars)
        for i in eachindex(param_seq.seq)
            llparam_map!(param_seq[i], pars[i], i)
        end
        
        return logpdf!(model, param_seq, observations, approx, reset_obs_state_iter_setup!) 
    end

    return loglikelihood
end

struct RandomWalkGammaInitialDistR0Prior{F}
    initial_dist::Gamma{F}
    randomwalkstddev::F
end

function Distributions.logpdf(rw::RandomWalkGammaInitialDistR0Prior{F}, R0s::AbstractVector) where F
    ll = logpdf(rw.initial_dist, R0s[1])
    for i in Iterators.drop(eachindex(R0s),1)
        ll += logpdf(Normal(R0s[i-1], rw.randomwalkstddev), R0s[i])
    end
    return ll
end

function makeprior(config)
    if config["inference"]["prior_parameters"]["R_0"]["type"]=="random_walk_gamma_initial_dist"
        init_dist = Gamma(config["inference"]["prior_parameters"]["R_0"]["shape"], 
                          config["inference"]["prior_parameters"]["R_0"]["scale"])
        sigma = config["inference"]["prior_parameters"]["R_0"]["sigma"]
        R0prior = RandomWalkGammaInitialDistR0Prior(init_dist, sigma)
        prior_logpdf = (params) -> begin
            val = logpdf(R0prior, params)
            return val
        end
    elseif config["inference"]["prior_parameters"]["R_0"]["type"]=="gaussian_processes"
        if config["inference"]["prior_parameters"]["R_0"]["covariance_function"]=="exponential"
            cov_fun = GP.ExponentialCovarianceFunction(
                config["inference"]["prior_parameters"]["R_0"]["sigma"]^2,
                config["inference"]["prior_parameters"]["R_0"]["ell"]
            )
            timestamps = Matrix(reshape(Float64.(config["model"]["stateprocess"]["params"]["timestamps"]), 1, :))
            mu = config["inference"]["prior_parameters"]["R_0"]["mu"]
            R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
        elseif config["inference"]["prior_parameters"]["R_0"]["covariance_function"]=="squared_exponential"
            cov_fun = GP.SquaredExponentialCovarianceFunction(
                config["inference"]["prior_parameters"]["R_0"]["sigma"]^2,
                config["inference"]["prior_parameters"]["R_0"]["ell"]
            )
            timestamps = Matrix(reshape(Float64.(config["model"]["stateprocess"]["params"]["timestamps"]), 1, :))
            mu = config["inference"]["prior_parameters"]["R_0"]["mu"]
            R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
        else 
            error("Unknown covariance function in config")
        end
        if config["inference"]["prior_parameters"]["R_0"]["transform"]=="log"
            cache = zeros(Float64, length(timestamps))
            gpmemcache = GP.gp_logpdf_memcache(R0prior, cache)
            prior_logpdf = (params) -> begin
                if any(p -> p <= zero(p), params)
                    return -Inf
                end
                for i in eachindex(params)
                    cache[i] = log(params[i])
                end
                val = GP.logpdf(R0prior, cache, gpmemcache)
                val -= sum(cache)
                return val
            end
        elseif config["inference"]["prior_parameters"]["R_0"]["transform"]=="none"
            cache = zeros(Float64, length(timestamps))
            gpmemcache = GP.gp_logpdf_memcache(R0prior, cache)
            prior_logpdf = (params) -> begin
                for i in eachindex(params)
                    cache[i] = params[i]
                end
                val = GP.logpdf(R0prior, cache, gpmemcache)
                return val
            end
        else
            error("Unkown R_0 prior specification, expected \"log\" or \"none\", got $(config["inference"]["prior_parameters"]["R_0"]["transform"])")
        end
    else
        error("Unknown prior R0 specififcation")
    end
    return prior_logpdf
end
