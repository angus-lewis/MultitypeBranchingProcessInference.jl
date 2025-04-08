using YAML
using MultitypeBranchingProcessInference
using Random 

include("./src/forecasting.jl")

function main(argv)
    argc = length(argv)
    if argc != 1
        error("inference.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    raw_observations = read_observations(joinpath(pwd(), config["data"]["filename"]))
    t = config["data"]["first_observation_time"] .+ (0:(length(raw_observations)-1))
    observations = Observations(t, raw_observations)

    dow_prior = config["model"]["fixed_parameters"]["observation_model"]["day_of_week_effect_prior"]
    dow_effect = estimate_dow_effect(vcat(raw_observations...), 0, dow_prior, length(dow_prior))

    epidemicmodel = makemodel(
        config["model"]["fixed_parameters"]["T_E"],
        config["model"]["fixed_parameters"]["T_I"],
        dow_effect,
        config["model"]["fixed_parameters"]["observation_model"]["variance"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        config["model"]["inferred_parameters"]["R_0"]["initial_values"],
        config["model"]["fixed_parameters"]["E_state_count"],
        config["model"]["fixed_parameters"]["I_state_count"],
        config["model"]["fixed_parameters"]["initial_state"],
        observations,
        config["model"]["fixed_parameters"]["immigration_rate"]=="nothing" ? nothing : config["model"]["fixed_parameters"]["immigration_rate"],
        config["model"]["fixed_parameters"]["notification_rate"]=="nothing" ? nothing : config["model"]["fixed_parameters"]["notification_rate"],
        config["model"]["fixed_parameters"]["observation_probability"],
    )

    prior_dist = makeprior(
        config["model"]["inferred_parameters"]["R_0"]["prior"]["mu"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_type"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_parameters"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["transform"]
    )

    proposal_distribuion = makeforecastingproposal(config)
    mh_rng, mh_config = makemhconfig(config)

    MetropolisHastings.skip_binary_array_file_header(mh_config.model_info_io, 2)

    @time nsamples = MetropolisHastings.metropolis_hastings(
        mh_rng, epidemicmodel, prior_dist, proposal_distribuion, mh_config,
    )

    MetropolisHastings.write_binary_array_file_header(
        mh_config.model_info_io, 
        (length(epidemicmodel.info_cache), nsamples)
    )

    if mh_config.model_info_io !== devnull && mh_config.model_info_io !== stdout
        close(mh_config.model_info_io)
    end

    println()
    return
end

main(ARGS)