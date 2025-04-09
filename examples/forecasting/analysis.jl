using StatsPlots
using MCMCChains
using YAML
using MultitypeBranchingProcessInference

include("src/forecasting.jl")

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
    "data/dow_expcovfn_vic_covid_kalman_param_samples.f64_array.bin";
    # "data/_dow_expcovfn_vic_covid_kalman_param_samples.f64_array.bin";
]
datasetnames = ["dow"]
nburnin = 231000
paramnames = vcat(
    [:L; :L0; :k; :x0],
    [Symbol("R_0_$i") for i in 1:11], 
    [:LL]
)

samples = read_datasets(filenames, datasetnames, nburnin, paramnames)

info_filenames = [
    "data/_dow_expcovfn_vic_covid_kalman.model_info.f64_array.bin"
]
ntypes = 5
info_paramnames = vcat(
    [Symbol("State_$i") for i in 1:ntypes], 
    [Symbol("Cov(State_$i, State_$j)") for i in 1:ntypes, j in 1:ntypes][:], 
)

model_info = read_datasets(info_filenames, datasetnames, nburnin, info_paramnames)

function evalmeanfun(samps, times)
    out = zeros(length(times), size(samps,2))
    for i in axes(samps,2)
        fun = Logistic4(samps[1:4,i]...)
        out[:,i] .= fun.(times)
    end
    return out
end

plot(samples[datasetnames[1]])
for i in Iterators.drop(datasetnames, 1)
    plot!(samples[i][paramnames[1:end]])
end
plot!()

plot(model_info[datasetnames[1]][info_paramnames[1:end-1]])
# for i in Iterators.drop(datasetnames, 1)
#     plot!(model_info[i][info_paramnames[1:end-1]])
# end
plot!()

means = evalmeanfun(samples["dow"].value.data.parent', 0:7:72)

histogram(exp.(means[:,1:10:end] .- 0.4)', alpha=0.2)
histogram!(samples["dow"].value.data.parent[1:10:end,5:end-1], alpha=0.2)

plot(exp.(means[:,1:100:end] .- 0.4), alpha=0.05, color=:blue, label=false)
