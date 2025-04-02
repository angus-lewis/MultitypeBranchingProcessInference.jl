# A SEIR model with fixed T_E, T_I, observation probability=1, with time-varying
# observation model (7-day period), R0 inferred from data with a Gaussian 
# Process prior distribution and no immigration.

function makemodel(T_E, T_I, observation_scale_factors, observation_variance, R0changepoints, R0s,
    E_state_count=1, I_state_count=1, initial_state=[1; zeros(Int, E_state_count+I_state_count-1+2)],
)
    length(observation_scale_factors)==OBS_MODEL_PERIOD || error("Expected $OBS_MODEL_PERIOD observation scalings")
    length(R0s)==length(R0changepoints) || error("Must have the same number of R0 initial values as changepoints")

    delta, lambda, beta = convertseirparamstorates(first(R0s), T_E, T_I, E_state_count, I_state_count)
    seir = SEIR(
        E_state_count,
        I_state_count,
        beta, 
        delta, 
        lambda, 
        OBSERVTION_PROBABILITY, 
        IMMIGRATION_RATE, 
        initial_state,
    )

    seir_param_vec = MTBPParams{paramtype(seir), Vector{paramtype(seir)}}[]
    seir_param_seq = MTBPParamsSequence(seir_param_vec)
    immigration_vec = fill(IMMIGRATION_RATE, E_state_count+I_state_count)
    for i in eachindex(R0s)
        seir_params = MTBPParams(seir)
        seir_params.time = R0changepoints[i]
        param_map!(seir_params, E_state_count, I_state_count, (R0s[i], T_E, T_I), immigration_vec)
        push!(seir_param_seq.seq, seir_params)
    end

    observation_matrices = Matrix[]
    for i in 1:OBS_MODEL_PERIOD
        obs_operator = zeros(1, getntypes(seir))
        obs_operator[obs_state_idx(seir)] = observation_scale_factors[i]
        push!(observation_matrices, obs_operator)
    end

    observation_model = LinearGaussianObservationModel(
        observation_matrices[1], zeros(1), fill(observation_variance, 1, 1)
    )

    model = StateSpaceModel(seir, observation_model)
    return InferenceModel(model, seir_param_seq, observation_matrices)
end

function makeparammap(E_state_count, I_state_count, immigration_vec, T_E, T_I)
    llparam_map! = (mtbpparams, R0) -> begin
        return param_map!(
            mtbpparams, 
            E_state_count, 
            I_state_count, 
            (R0, T_E, T_I), 
            immigration_vec, 
        )
    end
    return llparam_map!
end

function makeapprox(inferencemodel)
    approx = MTBPKalmanFilterApproximation(inferencemodel.model)
    return approx
end

function makeloglikelihood(T_E, T_I, observations, inferencemodel, setuprng, E_state_count, I_state_count, approx)
    
    immigration_vec = fill(IMMIGRATION_RATE, E_state_count+I_state_count)
    
    llparam_map! = makeparammap(E_state_count, I_state_count, immigration_vec, T_E, T_I)

    day_of_week_itersetup! = (f, model, dt, observation, iteration, use_prev_iter_params) -> begin
        day_of_week =  (iteration-1)%OBS_MODEL_PERIOD + 1
        observation_matrix = inferencemodel.observation_matrices[day_of_week]
        inferencemodel.model.observation_model.obs_map .= observation_matrix
        return 
    end

    inferenceitersetup! = (args...) -> begin
        reset_obs_state_iter_setup!(args...)
        day_of_week_itersetup!(args...)
        return 
    end

    loglikelihood = (params) -> begin # function loglikelihood(params)
        for i in eachindex(inferencemodel.paramseq.seq)
            llparam_map!(inferencemodel.paramseq[i], params[i])
        end
        ll = logpdf!(
            inferencemodel.model, inferencemodel.paramseq, 
            observations, approx, inferenceitersetup!
        )
        return ll
    end

    return loglikelihood
end
