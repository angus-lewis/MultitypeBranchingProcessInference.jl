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

    epidemicmodel = makemodel(
        config["model"]["fixed_parameters"]["T_E"],
        config["model"]["fixed_parameters"]["T_I"],
        config["model"]["fixed_parameters"]["observation_model"]["scale_factors"],
        config["model"]["fixed_parameters"]["observation_model"]["variance"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        config["model"]["inferred_parameters"]["R_0"]["initial_values"],
        config["model"]["fixed_parameters"]["E_state_count"],
        config["model"]["fixed_parameters"]["I_state_count"],
        config["model"]["fixed_parameters"]["initial_state"],
        observations,
    )
    seir = epidemicmodel.model.stateprocess

    r0prior = makepriordist(
        config["model"]["inferred_parameters"]["R_0"]["prior"]["mu"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_type"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_parameters"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
    )

    forecastrng = makerng(config["forecast"]["seed"])
    forecasttimes = Matrix{Float64}(reshape(config["forecast"]["times"], 1, :))
    forecastR0mu = config["forecast"]["R_0"]["mu"]

    nforecastsims = config["forecast"]["nsims"]
    
    r0sampled, statessampled = sample_posteriors(forecastrng, nforecastsims, config, getntypes(seir))

    transform = config["model"]["inferred_parameters"]["R_0"]["prior"]["transform"]
    r0forecast = forecast_R0(forecastrng, r0prior, r0sampled, forecasttimes, forecastR0mu, transform)
    
    cases = forecastcases(forecastrng, epidemicmodel, last(observations).time, forecasttimes, statessampled, r0forecast)

    doweffect = config["model"]["fixed_parameters"]["observation_model"]["scale_factors"]
    apply_dow_effect!(cases, forecasttimes, doweffect)

    cases .= round.(cases)
    
    write_forecasts(config["forecast"]["outfilename"], forecasttimes, cases)
    return raw_observations, t, cases, forecasttimes, r0forecast, statessampled, r0sampled
end

raw_observations, t, cases, forecasttimes, r0forecasts, statesampled, r0sampled = main(ARGS)