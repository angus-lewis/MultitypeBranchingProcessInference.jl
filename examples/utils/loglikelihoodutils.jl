function reset_obs_state_iter_setup!(
    f::HybridFilterApproximation,
    model, dt, observation, iteration, use_prev_iter_params,
)
    return 
end
function reset_obs_state_iter_setup!(
    f::MTBPKalmanFilterApproximation,
    model, dt, observation, iteration, use_prev_iter_params,
)
    reset_idx = obs_state_idx(model.stateprocess)
    kf = f.kalmanfilter
    kf.state_estimate[reset_idx] = zero(eltype(kf.state_estimate))
    kf.state_estimate_covariance[:, reset_idx] .= zero(eltype(kf.state_estimate_covariance))
    kf.state_estimate_covariance[reset_idx, :] .= zero(eltype(kf.state_estimate_covariance))
    return 
end
function reset_obs_state_iter_setup!(
    f::ParticleFilterApproximation,
    model, dt, observation, iteration, use_prev_iter_params,
)
    reset_idx = obs_state_idx(model.stateprocess)
    for particle in f.store.store
        particle[reset_idx] = zero(eltype(particle))
    end
    return 
end

function convertseirparamstorates(R_0, T_E, T_I, E_state_count, I_state_count)
    # rate of symptom onset
    delta = E_state_count/T_E
    # rate of recovery
    lambda = I_state_count/T_I
    # rate of infection
    beta = R_0/T_I
    return delta, lambda, beta
end

function param_map!(
    mtbpparams, 
    E_state_count, 
    I_state_count, 
    seir_params, # R_0, T_E, T_I 
    immigration, 
    convert_to_rates=true,
)
    R_0, T_E, T_I = seir_params
    if convert_to_rates
        delta, lambda, beta = convertseirparamstorates(R_0, T_E, T_I, E_state_count, I_state_count)
    else 
        delta, lambda, beta = seir_params
    end
    # exposed individuals progress to infectious at rate delta
    exposed_states = 1:E_state_count
    for i in exposed_states
        mtbpparams.rates[i] = delta
    end
    # Note: infection events are either observed or unobserved
    # with a fixed probability. Hence the cdfs of exposed progeny 
    # events is fixed and does not need to be updated

    # infectious individuals create infections at rate beta and recover at rate lambda
    infectious_states = (E_state_count+1):(E_state_count+I_state_count)
    for i in infectious_states
        mtbpparams.rates[i] = beta+lambda
    end
    mtbpparams.rates[end-1] = zero(eltype(mtbpparams.rates))
    mtbpparams.rates[end] = sum(immigration)

    p = beta/(beta+lambda)
    for i in infectious_states
        mtbpparams.cdfs[i][1] = p
        mtbpparams.cdfs[i][2] = one(eltype(mtbpparams.rates))
    end

    if iszero(mtbpparams.rates[end])
        mtbpparams.cdfs[end] .= range(zero(eltype(mtbpparams.cdfs[end])), one(eltype(mtbpparams.cdfs[end])), length(immigration))
    else
        mtbpparams.cdfs[end] .= cumsum(immigration)
        mtbpparams.cdfs[end] ./= mtbpparams.cdfs[end][end]
    end
    return mtbpparams
end