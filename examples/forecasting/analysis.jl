using StatsPlots
using MCMCChains
using YAML
using MultitypeBranchingProcessInference

function read_datasets(filenames, datasetnames, nburnin, paramnames)
    fileid = 1
    chains = Dict{String, Chains}()
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        dataset = samples[:, (nburnin+1):end]
        
        chain = Chains(dataset', paramnames)
        open(joinpath(pwd(), "$(filename).summary.txt"), "w") do io
            display(TextDisplay(io), chain)
        end
        chains[datasetname] = chain
        fileid += 1
    end
    return chains
end

filenames = [
    "data/dow_expcovfn_vic_covid_kalman_param_samples.f64_array.bin"
]
datasetnames = ["dow"]
nburnin = 0
paramnames = vcat(
    [Symbol("R_0_$i") for i in 1:11], 
    [:LL]
)

samples = read_datasets(filenames, datasetnames, nburnin, paramnames)

info_filenames = [
    "data/dow_expcovfn_vic_covid_kalman.model_info.f64_array.bin"
]
ntypes = 4
info_paramnames = vcat(
    [Symbol("State_$i") for i in 1:ntypes], 
    [Symbol("Cov(State_$i, State_$j)") for i in 1:ntypes, j in 1:ntypes][:], 
)

model_info = read_datasets(info_filenames, datasetnames, nburnin, info_paramnames)

plot(samples[datasetnames[1]][paramnames[1:end-1]])
# for i in Iterators.drop(datasetnames, 1)
#     plot!(samples[i][paramnames[1:end-1]])
# end
plot!()

plot(model_info[datasetnames[1]][info_paramnames[1:end-1]])
# for i in Iterators.drop(datasetnames, 1)
#     plot!(model_info[i][info_paramnames[1:end-1]])
# end
plot!()

