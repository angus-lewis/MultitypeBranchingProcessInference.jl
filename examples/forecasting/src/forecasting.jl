using LinearAlgebra
using Random

using MultitypeBranchingProcessInference

include("../../utils/gaussianprocesses.jl")
include("../../utils/loglikelihoodutils.jl")
include("../../utils/generalutils.jl")
include("../../utils/mhutils.jl")

const OBSERVTION_PROBABILITY = 1.0
const IMMIGRATION_RATE = 0.0
const OBS_MODEL_PERIOD = 7

struct InferenceModel{M<:StateSpaceModel, 
        P<:MTBPParamsSequence,
        O<:AbstractVector{<:Matrix}}
    model::M
    paramseq::P
    observation_matrices::O
end

include("model.jl")
include("prior.jl")
