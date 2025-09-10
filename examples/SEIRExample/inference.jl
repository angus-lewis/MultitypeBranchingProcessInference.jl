using YAML
using Distributions
using Random 
using LinearAlgebra
using DelimitedFiles

using MultitypeBranchingProcessInference

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

loglikelihood = makeloglikelihood(model, params_seq, observations, config)

struct SSMWrapper{T<:StateSpaceModel}
    model::T
end

function Distributions.logpdf(model::SSMWrapper, params)
    return loglikelihood(model.model, params)
end

# prior logpdf function 
prior_logpdf = makeprior(config)

struct PriorType end

function Distributions.logpdf(prior::PriorType, params)
    return prior_logpdf(params)
end


proposal_distribuion = makeproposal(config)

mh_rng, mh_config, samples_file_io, info_file_io, model_info_file_io = makemhconfig(config)

# run mcmc
MetropolisHastings.skip_binary_array_file_header(samples_file_io, 2)
@time mh_nsamples, _, _ = MetropolisHastings.metropolis_hastings(
    mh_rng, SSMWrapper(model), PriorType(), proposal_distribuion, mh_config, samples_file_io, info_file_io, model_info_file_io
)
MetropolisHastings.write_binary_array_file_header(samples_file_io, (mh_config.nparams+1, mh_nsamples))
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

