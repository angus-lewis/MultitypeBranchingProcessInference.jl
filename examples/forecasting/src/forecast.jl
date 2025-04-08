function forecast_R0(rng, prior_gp, posterior_samples, forecasttimes, forecastpriormean, transform)
    if transform == "log"
        posterior_samples .= log.(posterior_samples)
    end
    
    forecastsamples = rand(rng, prior_gp, posterior_samples, forecasttimes, forecastpriormean)
    
    if transform == "log"
        forecastsamples .= exp.(forecastsamples)
        posterior_samples .= exp.(posterior_samples)
    end

    return forecastsamples
end

function ensure_symmetric!(array)
    for i in axes(array, 2)
        for j in Iterators.drop(axes(array, 1), i)
            array[j,i] += array[i,j]
            array[j,i] /= 2
            array[i,j] = array[j,i]
        end
    end
    return array
end

function forecastcases(rng, model, lastdatatime, forecasttimes, initialstatesamples, r0forecasts)
    @assert size(r0forecasts,2) == size(initialstatesamples,2)
    @assert length(forecasttimes) == size(r0forecasts,1)
    @assert getntypes(model.model.stateprocess) == size(initialstatesamples,1)
    @assert lastdatatime <= first(forecasttimes) "forecast must be after last observation  (last observations = $(lastdatatime) and first forecast time = $(first(forecasttimes)))"
    
    seir = model.model.stateprocess
    mtbpparams = MTBPParams(seir)
    moments_operator = MTBPMomentsOperator(getntypes(seir))

    forecaststate = zeros(length(forecasttimes), size(r0forecasts, 2))
    prevstatesamp = zeros(paramtype(moments_operator), getntypes(seir))
    statesamp = zeros(paramtype(moments_operator), getntypes(seir))
    statemean = zeros(paramtype(moments_operator), getntypes(seir))
    statecov = zeros(paramtype(moments_operator), getntypes(seir), getntypes(seir))

    for n in axes(r0forecasts, 2)
        prevstatesamp .= initialstatesamples[:,n]
        prevstatesamp .= round.(initialstatesamples[:,n])
        prevstatesamp[model.observed_state_idx] = 0
        prevt = lastdatatime

        for i in axes(r0forecasts, 1)
            R0 = r0forecasts[i,n]
            dt = forecasttimes[i] - prevt
            prevt = forecasttimes[i]
            
            llparam_map!(mtbpparams, R0, model)
            setparams!(seir, mtbpparams)
            
            moments!(moments_operator, seir, dt)
            mean!(statemean, moments_operator, prevstatesamp)
            variance_covariance!(statecov, moments_operator, prevstatesamp)
            ensure_symmetric!(statecov)

            Z = MvNormal(statemean[1:end-1], statecov[1:end-1,1:end-1])
            @views statesamp[1:end-1] .= rand(rng, Z)
            statesamp[end] = zero(eltype(statesamp))
            
            statesamp[statesamp .< zero(eltype(statesamp))] .= zero(eltype(statesamp))
            
            forecaststate[i, n] = statesamp[model.observed_state_idx]

            prevstatesamp .= statesamp
            prevstatesamp[model.observed_state_idx] = zero(eltype(prevstatesamp))
        end
    end
    return forecaststate
end

function read_samples(filename, nburnin)
    samples = open(joinpath(pwd(), filename), "r") do io
        MetropolisHastings.read_binary_array_file(io)
    end
    samples = samples[:, (nburnin+1):end]
    return samples
end

function sample_posteriors(rng, nsamples, config, ntypes)
    # read data
    r0 = read_samples(
        config["forecast"]["R_0"]["samples"]["filename"], 
        config["forecast"]["R_0"]["samples"]["nburnin"],
    )
    state_params = read_samples(
        config["forecast"]["state"]["samples"]["filename"], 
        config["forecast"]["state"]["samples"]["nburnin"],
    )
    @assert size(state_params, 2) == size(r0, 2)

    # sample
    sample_idx = rand(rng, axes(r0, 2), nsamples)

    # sample R0
    # ignore LL values in the last row
    r0samples = r0[1:end-1, sample_idx]

    # sample state params
    statemeanssampled = state_params[1:ntypes, sample_idx]
    statecovsampled = state_params[ntypes .+ (1:ntypes^2), sample_idx]

    # samples states
    statessampled = similar(statemeanssampled)
    isimmigrationmodel = config["immigration_rate"]!="nothing"
    for i in axes(statessampled, 2)
        @views mu = statemeanssampled[:,i]
        @views cov = reshape(statecovsampled[:,i], ntypes, ntypes)
        ensure_symmetric!(cov)

        # ignore last element as it is immigration
        @views Z = MvNormal(mu[1:end-isimmigrationmodel], cov[1:end-isimmigrationmodel, 1:end-isimmigrationmodel])
        statessampled[1:end-1,i] .= rand(rng, Z)
    end
    statessampled[end,:] .= 0

    statessampled .= round.(statessampled)

    return r0samples, statessampled
end

function apply_dow_effect!(cases, forecasttimes, doweffect)
    period = length(doweffect)
    for j in axes(cases, 2)
        for i in eachindex(forecasttimes)
            day_idx = Int((forecasttimes[i]-1)%period + 1)
            cases[i,j] *= doweffect[day_idx]
        end
    end
    return 
end

function write_forecasts(filename, times, cases)
    @assert size(cases, 1)==length(times)
    open(filename, "w") do io
        print(io, "time")
        for i in axes(cases, 2)
            print(io, ",cases_$i")
        end
        print(io, "\n")

        for t in eachindex(times)
            print(io, "$(times[t])")
            for i in axes(cases, 2)
                print(io, ",$(Int(cases[t,i]))")
            end
            print(io, "\n")
        end
    end
    return 
end