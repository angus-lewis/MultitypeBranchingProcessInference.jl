using LinearAlgebra
using Random
using Distributions

import Random: rand

using MultitypeBranchingProcessInference

import MultitypeBranchingProcessInference.MetropolisHastings: adapt!, setstate!

include("../../utils/gaussianprocesses.jl")
include("../../utils/loglikelihoodutils.jl")
include("../../utils/generalutils.jl")
include("../../utils/mhutils.jl")

const OBSERVTION_PROBABILITY = 1.0
const IMMIGRATION_RATE = 0.0

include("model.jl")
include("prior.jl")
include("proposal.jl")
include("forecast.jl")
