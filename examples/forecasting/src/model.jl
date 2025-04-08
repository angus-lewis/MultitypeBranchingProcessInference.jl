# A SEIR model with fixed T_E, T_I, observation probability=1, with time-varying
# observation model (7-day period), R0 inferred from data with a Gaussian 
# Process prior distribution and no immigration.

struct EpidemicModel{M<:StateSpaceModel, F<:AbstractFloat, FN<:Union{F,Nothing}, S<:MTBPParamsSequence, 
    O<:Observations, K<:MTBPKalmanFilterApproximation, I<:Union{Vector{F}, Nothing}}
    model::M
    T_E::F
    T_I::F
    notification_rate::FN
    observed_state_idx::Int
    paramseq::S
    E_state_count::Int
    I_state_count::Int
    dow_effect::Array{Matrix{F}, 1}
    immigration::I
    observations::O
    kfapprox::K
    info_cache::Vector{F}
end

function makeapprox(model)
    approx = MTBPKalmanFilterApproximation(model)
    return approx
end

function makemodel(T_E, T_I, observation_scale_factors, observation_variance, R0changepoints, R0s,
    E_state_count, I_state_count, initial_state, observations, immigration_vec, notification_rate, observation_probability,
)
    @assert length(R0s)==length(R0changepoints) "Must have the same number of R0 initial values as changepoints"
    
    isnotificationmodel = notification_rate !== nothing
    
    delta, lambda, beta = convertseirparamstorates(first(R0s), T_E, T_I, E_state_count, I_state_count)
    seir = SEIR(
        E_state_count,
        I_state_count,
        beta, 
        delta, 
        lambda, 
        observation_probability,
        notification_rate,
        immigration_vec, 
        initial_state,
    )

    seir_param_vec = MTBPParams{paramtype(seir), Vector{paramtype(seir)}}[]
    seir_param_seq = MTBPParamsSequence(seir_param_vec)
    for i in eachindex(R0s)
        seir_params = MTBPParams(seir)
        seir_params.time = R0changepoints[i]
        param_map!(seir_params, E_state_count, I_state_count, (R0s[i], T_E, T_I), immigration_vec)
        push!(seir_param_seq.seq, seir_params)
    end

    obs_state_idx = E_state_count + I_state_count + 1 + isnotificationmodel
    observation_matrices = Matrix{Float64}[]
    for i in eachindex(observation_scale_factors)
        obs_operator = zeros(1, getntypes(seir))
        obs_operator[obs_state_idx] = observation_scale_factors[i]
        push!(observation_matrices, obs_operator)
    end

    observation_model = LinearGaussianObservationModel(
        observation_matrices[1], zeros(1), fill(observation_variance, 1, 1)
    )

    model = StateSpaceModel(seir, observation_model)

    kfapprox = makeapprox(model)

    model_info_cache = zeros(
        eltype(kfapprox.kalmanfilter.state_estimate),
        length(kfapprox.kalmanfilter.state_estimate) + length(kfapprox.kalmanfilter.state_estimate_covariance)
    )
    
    return EpidemicModel(
        model, 
        T_E, T_I, 
        notification_rate,
        obs_state_idx, 
        seir_param_seq, 
        E_state_count,
        I_state_count, 
        observation_matrices,
        immigration_vec,
        observations,
        kfapprox,
        model_info_cache,
    )
end

function llparam_map!(mtbpparams, R0, model)
    return param_map!(
        mtbpparams,
        model.E_state_count,
        model.I_state_count,
        (R0, model.T_E, model.T_I),
        model.immigration,
    )
end

function Distributions.logpdf(model::EpidemicModel, params)
    for i in eachindex(model.paramseq.seq)
        llparam_map!(model.paramseq[i], params[i], model)
    end

    day_of_week_itersetup! = (f, statespacemodel, dt, observation, iteration, use_prev_iter_params) -> begin
        day_of_week =  (iteration-1)%length(model.dow_effect) + 1
        observation_matrix = model.dow_effect[day_of_week]
        statespacemodel.observation_model.obs_map .= observation_matrix
        return 
    end

    reset_idx = model.E_state_count + model.I_state_count + 1 + (model.notification_rate!==nothing)
    reset_obs_state_iter_setup! = (f, ssmodel, dt, observation, iteration, use_prev_iter_params) -> begin
        reset_state_iter_setup!(f, ssmodel, dt, observation, iteration, use_prev_iter_params, reset_idx)
        return
    end

    inferenceitersetup! = (args...) -> begin
        reset_obs_state_iter_setup!(args...)
        day_of_week_itersetup!(args...)
        return 
    end

    ll = logpdf!(
        model.model, model.paramseq, 
        model.observations, model.kfapprox, inferenceitersetup!
    )
    return ll
end

function MetropolisHastings.write_model_info(io::IO, model::EpidemicModel, isacc::Bool)
    if isacc
        offset = 0
        for i in eachindex(model.kfapprox.kalmanfilter.state_estimate)
            model.info_cache[offset + i] = model.kfapprox.kalmanfilter.state_estimate[i]
        end
        offset += length(model.kfapprox.kalmanfilter.state_estimate)
        for i in eachindex(model.kfapprox.kalmanfilter.state_estimate_covariance)
            model.info_cache[offset + i] = model.kfapprox.kalmanfilter.state_estimate_covariance[i]
        end
    end
    write(io, model.info_cache)
    return
end
