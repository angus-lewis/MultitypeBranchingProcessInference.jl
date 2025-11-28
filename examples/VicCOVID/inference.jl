using YAML
using Random
using LinearAlgebra
using MultitypeBranchingProcessInference
using Distributions

import MultitypeBranchingProcessInference.MetropolisHastings.write_model_info

include("./utils/config.jl")

argv = ARGS

argc = length(argv)
if argc != 1
    error("inference.jl program expects 1 argument \
            \n    - config file name.")
end

config = YAML.load_file(joinpath(pwd(), argv[1]))

setenvironment!(config)

raw_observations = read_observations(joinpath(pwd(), config["inference"]["data"]["filename"]))
t = config["inference"]["data"]["first_observation_time"] .+ (0:(length(raw_observations)-1))
observations = Observations(t, raw_observations)

loglikelihood, ssm_model = makeloglikelihood(observations, config)

function Distributions.logpdf(model::SSMWrapper, params)
    return loglikelihood(model, params)
end

prior_logpdf = makeprior(config)

struct PriorWrapper end

function Distributions.logpdf(prior::PriorWrapper, params)
    return prior_logpdf(params)
end

proposal_distribuion = makeproposal(config)

function MultitypeBranchingProcessInference.MetropolisHastings.write_model_info(io::IO, ssm::SSMWrapper, samples_count, isacc::Bool, thin)
    if isacc
        # store until the next time the proposal is accepted
        ssm.prev_info_cache .= ssm.info_cache
    end
    if (samples_count-1)%thin == 0
        write(io, ssm.prev_info_cache)
    end
    return 
end

mh_rng, mh_config, samples_out, info_out, model_info_out = makemhconfig(config)

MetropolisHastings.skip_binary_array_file_header(samples_out, 2)
MetropolisHastings.skip_binary_array_file_header(model_info_out, 2)

# run mcmc
@time mh_nsamples, _, _ = MetropolisHastings.metropolis_hastings(
    mh_rng, ssm_model, PriorWrapper(), proposal_distribuion, mh_config, samples_out, info_out, model_info_out
)

MetropolisHastings.write_binary_array_file_header(
    samples_out, (mh_config.nparams+1, mh_nsamples)
)
MetropolisHastings.write_binary_array_file_header(
    model_info_out, (length(ssm_model.info_cache), mh_nsamples÷mh_config.model_info_thin)
)

if samples_out isa IO && samples_out !== stdout
    close(samples_out)
end
if info_out isa IO && info_out !== stdout
    close(info_out)
end
if model_info_out isa IO && model_info_out !== stdout
    close(model_info_out)
end
println()
