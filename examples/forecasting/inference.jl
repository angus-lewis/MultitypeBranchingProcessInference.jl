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

    inferencemodel = makemodel(
        config["model"]["fixed_parameters"]["T_E"],
        config["model"]["fixed_parameters"]["T_I"],
        config["model"]["fixed_parameters"]["observation_model"]["scale_factors"],
        config["model"]["fixed_parameters"]["observation_model"]["variance"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        config["model"]["inferred_parameters"]["R_0"]["initial_values"],
        config["model"]["fixed_parameters"]["E_state_count"],
        config["model"]["fixed_parameters"]["I_state_count"],
        config["model"]["fixed_parameters"]["initial_state"]
    )

    prior_logpdf = makeprior(
        config["model"]["inferred_parameters"]["R_0"]["prior"]["mu"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_type"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["covariance_function_parameters"],
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        config["model"]["inferred_parameters"]["R_0"]["prior"]["transform"]
    )

    kfapprox = makeapprox(inferencemodel)

    loglikelihood = makeloglikelihood(
        config["model"]["fixed_parameters"]["T_E"],
        config["model"]["fixed_parameters"]["T_I"],
        observations, 
        inferencemodel,
        makerng(config["model"]["inferred_parameters"]["initial_state"]["seed"]),
        config["model"]["fixed_parameters"]["E_state_count"],
        config["model"]["fixed_parameters"]["I_state_count"], 
        kfapprox
    )
    proposal_distribuion = makeproposal(config)
    mh_rng, mh_config = makemhconfig(config)

    @time nsamples = MetropolisHastings.metropolis_hastings(
        mh_rng, loglikelihood, prior_logpdf, proposal_distribuion, mh_config,
    )
    println()
    return
end

main(ARGS)