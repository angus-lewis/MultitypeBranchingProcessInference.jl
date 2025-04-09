using YAML
using MultitypeBranchingProcessInference
using StatsPlots

include("./src/forecasting.jl")

function main(argv)
    argc = length(argv)
    if argc != 1
        error("forecast.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    raw_observations = read_observations(joinpath(pwd(), config["data"]["filename"]))
    t = config["data"]["first_observation_time"] .+ (0:(length(raw_observations)-1))
    observations = Observations(t, raw_observations)

    dow_prior = config["model"]["fixed_parameters"]["observation_model"]["day_of_week_effect_prior"]
    dow_effect = estimate_dow_effect(vcat(raw_observations...), 0, dow_prior, length(dow_prior))

    changepoints = config["model"]["inferred_parameters"]["R_0"]["changepoints"]
    r0prior, r0meanmodel = makepriordist(
        config["model"]["inferred_parameters"]["R_0"]["prior"]["initial_mean_model_params"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_type"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_parameters"],
        changepoints,
    )
    model_param_offset = paramcount(r0meanmodel)

    initial_state = read_initial_state_distribution(config["model"]["inferred_parameters"]["initial_states"])

    epidemicmodel = makemodel(
        config["model"]["fixed_parameters"]["T_E"],
        config["model"]["fixed_parameters"]["T_I"],
        dow_effect,
        config["model"]["fixed_parameters"]["observation_model"]["variance"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        config["model"]["inferred_parameters"]["R_0"]["initial_values"],
        config["model"]["fixed_parameters"]["E_state_count"],
        config["model"]["fixed_parameters"]["I_state_count"],
        initial_state,
        observations,
        config["model"]["fixed_parameters"]["immigration_rate"]=="nothing" ? nothing : config["model"]["fixed_parameters"]["immigration_rate"],
        config["model"]["fixed_parameters"]["notification_rate"]=="nothing" ? nothing : config["model"]["fixed_parameters"]["notification_rate"],
        config["model"]["fixed_parameters"]["observation_probability"],
        model_param_offset,
    )
    seir = epidemicmodel.model.stateprocess

    forecastrng = makerng(config["forecast"]["seed"])
    forecasttimes = Matrix{Float64}(reshape(config["forecast"]["times"], 1, :))
    nforecastsims = config["forecast"]["nsims"]
    
    r0sampled, statessampled = sample_posteriors(forecastrng, nforecastsims, config, getntypes(seir))

    transform = config["model"]["inferred_parameters"]["R_0"]["prior"]["transform"]
    r0forecast = forecast_R0(forecastrng, r0prior, r0sampled, forecasttimes, r0meanmodel, transform)
    
    cases = forecastcases(forecastrng, epidemicmodel, last(observations).time, forecasttimes, statessampled, r0forecast)

    apply_dow_effect!(cases, forecasttimes, dow_effect)

    cases .= round.(cases)
    
    write_forecasts(config["forecast"]["outfilename"]["forecasts"], forecasttimes, cases)
    
    offset = paramcount(r0meanmodel)+1

    write_r0s(config["forecast"]["outfilename"]["r0paths"], changepoints, forecasttimes, r0sampled[offset:end,:], r0forecast)
    write_states(config["forecast"]["outfilename"]["statesamples"], statessampled)
    return raw_observations, t, cases, forecasttimes, r0forecast, statessampled, r0sampled
end

raw_observations, t, cases, forecasttimes, r0forecasts, statesampled, r0sampled = main(ARGS)

