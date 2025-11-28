include("../../utils/loglikelihoodutils.jl")
include("../../utils/generalutils.jl")
include("../../utils/mhutils.jl")

function pathtodailycases(path, cases_idx)
    cumulative_cases = [[state[cases_idx]] for state in path]
    daily_cases = diff(cumulative_cases)
    cases = [[cumulative_cases[1]]; daily_cases]
    return cases
end

function makemodel(config)
    seirconfig = config["model"]["stateprocess"]["params"]
    seir = SEIR(
        seirconfig["E_state_count"], seirconfig["I_state_count"],
        first(seirconfig["infection_rate"]), 
        first(seirconfig["exposed_stage_change_rate"]), 
        first(seirconfig["infectious_stage_change_rate"]), 
        seirconfig["observation_probability"], 
        nothing, # notification rate
        seirconfig["immigration_rate"]==="nothing" ? nothing : seirconfig["immigration_rate"],
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

    param_seq = MTBPParamsSequence(MTBPParams{paramtype(model), Vector{paramtype(model)}}[])
    mtbpparams = MTBPParams(seir)
    beta, lambda, delta = seirconfig["infection_rate"], seirconfig["infectious_stage_change_rate"], seirconfig["exposed_stage_change_rate"]
    immigration = seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"]
    param_map!(
        mtbpparams, 
        seirconfig["E_state_count"], seirconfig["I_state_count"], 
        (delta, lambda, beta), immigration, false
    )
    push!(param_seq.seq, mtbpparams)
    
    return model, param_seq
end

function makepriordists(config)
    cts_prior_dists = Any[
        Gamma(config["inference"]["prior_parameters"]["R_0"]["shape"], 
              config["inference"]["prior_parameters"]["R_0"]["scale"]),
    ]
    discrete_prior_dists = Any[]
    return cts_prior_dists, discrete_prior_dists
end

function makeprior(config)
    cts_prior_dists, discrete_prior_dists = makepriordists(config)
    
    cts_prior_dists = tuple(cts_prior_dists...)
    discrete_prior_dists = tuple(discrete_prior_dists...)

    function prior_logpdf(params) 
        val = zero(eltype(params))
        for i in eachindex(cts_prior_dists)
            val += logpdf(cts_prior_dists[i], params[i])
        end
        for i in eachindex(discrete_prior_dists)
            val += logpdf(discrete_prior_dists[i], round(Int, params[i+length(cts_prior_dists)]))
        end
        return val
    end
    return prior_logpdf
end

struct SSMWrapper{T<:StateSpaceModel, V<:AbstractVector}
    model::T
    info_cache::V
    prev_info_cache::V
end

function makeloglikelihood(model, param_seq, observations, config)
    if config["inference"]["likelihood_approx"]["method"] == "hybrid"
        pf_rng = makerng(config["inference"]["likelihood_approx"]["particle_filter"]["seed"])
        nparticles = config["inference"]["likelihood_approx"]["particle_filter"]["nparticles"]

        switch_rng = makerng(config["inference"]["likelihood_approx"]["switch"]["seed"])
        switch_threshold = config["inference"]["likelihood_approx"]["switch"]["threshold"]
        
        randomstatesidx = 1:(getntypes(model.stateprocess) - 1) # all but cumulative cases

        approx = HybridFilterApproximation(
            model, pf_rng, switch_rng, nparticles, switch_threshold, randomstatesidx
        )
    elseif config["inference"]["likelihood_approx"]["method"] == "particle_filter"
        pf_rng = makerng(config["inference"]["likelihood_approx"]["particle_filter"]["seed"])
        nparticles = config["inference"]["likelihood_approx"]["particle_filter"]["nparticles"]

        approx = ParticleFilterApproximation(model, pf_rng, nparticles)

        if "switch" in keys(config["inference"]["likelihood_approx"])
            @warn "Unused \"switch\" params in config with approximation method \"particle_filter\"."
        end
    elseif config["inference"]["likelihood_approx"]["method"] == "kalman_filter"
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
    function llparam_map!(mtbpparams, param)
        return param_map!(
            mtbpparams, seirconfig["E_state_count"], seirconfig["I_state_count"], 
            (only(param), seirconfig["E_state_count"]/seirconfig["exposed_stage_change_rate"], seirconfig["I_state_count"]/seirconfig["infectious_stage_change_rate"]), 
            seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"], true
        )
    end

    reset_idx = seirconfig["E_state_count"] + seirconfig["I_state_count"] + 1
    ntypes = (seirconfig["E_state_count"] + seirconfig["I_state_count"] + 1)
    n_obs = length(observations)

    if approx isa MTBPKalmanFilterApproximation
        kf_state_est_params_count = length(approx.kalmanfilter.state_estimate)+length(approx.kalmanfilter.state_estimate_covariance)
        state_params_cache = zeros(eltype(approx.kalmanfilter.state_estimate), kf_state_est_params_count*(n_obs+1))
    elseif approx isa ParticleFilterApproximation
        state_params_cache = zeros(Int, ntypes*(n_obs+1))
    else
        state_params_cache = Float64[]
    end

    ssm = SSMWrapper(model, state_params_cache, similar(state_params_cache))

    reset_obs_state_iter_setup! = (f, model, dt, observation, iteration, use_prev_iter_params) -> begin
        if f isa MTBPKalmanFilterApproximation
            offset = (iteration-1)*kf_state_est_params_count
            ssm.info_cache[offset .+ (1:length(f.kalmanfilter.state_estimate))] .= (
                f.kalmanfilter.state_estimate
            )
            offset += length(f.kalmanfilter.state_estimate)
            ssm.info_cache[offset .+ (1:length(f.kalmanfilter.state_estimate_covariance))] .= (
                f.kalmanfilter.state_estimate_covariance[:]
            )
        elseif f isa ParticleFilterApproximation
            offset = (iteration-1)*ntypes
            ssm.info_cache[offset .+ (1:ntypes)] .= f.store.store[1]
        end
        reset_state_iter_setup!(f, model, dt, observation, iteration, use_prev_iter_params, reset_idx)
        return
    end

    loglikelihood = (ssm_model::SSMWrapper, pars) -> begin # function loglikelihood(pars)
        llparam_map!(only(param_seq), pars)
        ll = logpdf!(ssm_model.model, param_seq, observations, approx, reset_obs_state_iter_setup!)
        if approx isa MTBPKalmanFilterApproximation
            offset = length(observations)*kf_state_est_params_count
            ssm_model.info_cache[offset .+ (1:length(approx.kalmanfilter.state_estimate))] .= (
                approx.kalmanfilter.state_estimate
            )
            offset += length(approx.kalmanfilter.state_estimate)
            ssm_model.info_cache[offset .+ (1:length(approx.kalmanfilter.state_estimate_covariance))] .= (
                approx.kalmanfilter.state_estimate_covariance[:]
            )
        elseif approx isa ParticleFilterApproximation
            offset = length(observations)*(seirconfig["E_state_count"] + seirconfig["I_state_count"] + 1)
            ssm_model.info_cache[offset .+ (1:ntypes)] .= approx.store.store[1]
        end
        return ll
    end
    return loglikelihood, ssm
end