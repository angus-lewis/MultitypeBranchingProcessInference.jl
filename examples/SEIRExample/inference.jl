using YAML
using Distributions
using Random 
using LinearAlgebra
using DelimitedFiles

using MultitypeBranchingProcessInference

import MultitypeBranchingProcessInference.MetropolisHastings.write_model_info

include("./utils/config.jl")
include("./utils/io.jl")

argv = ARGS
argc = length(argv)
if argc != 1
    error("inference.jl program expects 1 argument \
            \n    - config file name.")
end

config = YAML.load_file(joinpath(pwd(), argv[1]))

setenvironment!(config)

model, params_seq = makemodel(config)
N = config["model"]["stateprocess"]["params"]["E_state_count"]
M = config["model"]["stateprocess"]["params"]["I_state_count"]

# get data
path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
# only need daily cases
obs_state_idx = 3 # yucky
cases = pathtodailycases(path, obs_state_idx)
observations = Observations(t, cases)

loglikelihood, ssm_model = makeloglikelihood(model, params_seq, observations, config)

# define this as it is what is called in metropolis_hastings
function Distributions.logpdf(model::SSMWrapper, params)
    return loglikelihood(model, params)
end

# prior logpdf function 
prior_logpdf = makeprior(config)

struct PriorType end

function Distributions.logpdf(prior::PriorType, params)
    return prior_logpdf(params)
end

proposal_distribuion = makeproposal(config)

function MultitypeBranchingProcessInference.MetropolisHastings.write_model_info(io::IO, ssm::SSMWrapper, samples_count, isacc::Bool, thin)
    if isacc
        # store until the next time the proposal is accepted
        ssm.prev_info_cache .= ssm.info_cache
    end
    write(io, ssm.prev_info_cache)
    return 
end

mh_rng, mh_config, samples_file_io, info_file_io, model_info_file_io = makemhconfig(config)

# run mcmc
MetropolisHastings.skip_binary_array_file_header(samples_file_io, 2)
MetropolisHastings.skip_binary_array_file_header(model_info_file_io, 2)

@time mh_nsamples, _, _ = MetropolisHastings.metropolis_hastings(
    mh_rng, ssm_model, PriorType(), proposal_distribuion, mh_config, samples_file_io, info_file_io, model_info_file_io
)

MetropolisHastings.write_binary_array_file_header(samples_file_io, (mh_config.nparams+1, mh_nsamples))
MetropolisHastings.write_binary_array_file_header(model_info_file_io, (length(ssm_model.info_cache), mh_nsamples))

if samples_file_io isa IO && samples_file_io !== stdout
    close(samples_file_io)
end
if info_file_io isa IO && info_file_io !== stdout
    close(info_file_io)
end
if model_info_file_io isa IO && model_info_file_io !== stdout
    close(model_info_file_io)
end

println()
