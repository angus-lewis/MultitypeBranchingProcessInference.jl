using StatsPlots
using MCMCChains
using YAML
using MultitypeBranchingProcessInference

function read_datasets(filenames, datasetnames, nburnin, paramnames)
    datasets = Dict{String, Array{Float64,2}}()
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
    "data/8800fixedIC_expcovfn_vic_covid_kalman_param_samples.f64_array.bin";
    "data/1000_sim_IC_expcovfn_vic_covid_kalman_param_samples.f64_array.bin"
    "data/dow_expcovfn_vic_covid_kalman_param_samples.f64_array.bin"
]
datasetnames = ["8800fixed"; "1000sim"; "dow"]
nburnin = 0
paramnames = vcat([Symbol("R_0_$i") for i in 1:14], [:LL])

datasets = read_datasets(filenames, datasetnames, nburnin, paramnames)

plot(datasets[datasetnames[1]][paramnames[1:end-1]])
for i in Iterators.drop(datasetnames, 1)
    plot!(datasets[i][paramnames[1:end-1]])
end
plot!()

