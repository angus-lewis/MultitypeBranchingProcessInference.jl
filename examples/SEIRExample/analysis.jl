using StatsPlots
using MCMCChains
using YAML
using KernelDensity
using Distributions
using LaTeXStrings
using MultitypeBranchingProcessInference
using OrderedCollections

include("./utils/config.jl")
include("./utils/figs.jl")
include("./utils/io.jl")

function read_datasets(filenames, datasetnames, nburnin, paramnames)
    datasets = OrderedDict{String, Array{Float64,2}}()
    fileid = 1
    chains = OrderedDict{String, Chains}()
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        # drop last row (log likelihoods) and burn-in
        dataset = samples[1:end-1, (nburnin+1):end]
        chain = Chains(dataset', paramnames)
        open(joinpath(pwd(), "$(filename).summary.txt"), "w") do io
            display(TextDisplay(io), chain)
        end
        chains[datasetname] = chain
        fileid += 1
    end
    return chains
end

function maketraceplots(chains)
    plots = OrderedDict{String, Any}()
    p = nothing
    chainid = 1
    for (datasetname, chain) in chains
        if chainid == 1
            p = plot(chain; color=cmap(chainid), linestyle=smap(1))
        else
            plot!(p, chain; color=cmap(chainid), linestyle=smap(1))
        end
        q = plot(chain; color=cmap(chainid), linestyle=smap(1))
        plots[datasetname] = q
        chainid += 1
    end
    if "all" in keys(chains)
        error("Key clash. A samples file cannot have the name all")
    end
    plots["all"] = p
    return plots
end

function make1dposteriorpdf(chains, paramname, prior=nothing, ymax=1.7, legend=true)
    p = plot(xlabel=L"%$(string(paramname))", ylabel="Density", legend=legend)
    chainid = 1
    for (datasetname, chain) in chains
        density!(p, chain[paramname]; label=datasetname, linestyle=smap(1), color=cmap(chainid), linewidth=2, ylims=(0,ymax), xlims=(0,5), bandwidth=0.04)
        chainid += 1
    end
    yl, yh = ylims(p)
    chainid = 1
    for (datasetname, chain) in chains
        x = fill(mean(chain[paramname]), 2)
        y = [yl, yh]
        plot!(p, x, y; label=false, linestyle=smap(2), color=cmap(chainid), linewidth=2)
        chainid += 1
    end
    if prior!==nothing
        plot!(p, x->Distributions.pdf(prior, x); label="Prior", color=cmap(chainid), linestyle=smap(3), linewidth=2)
    end
    return p
end

 function main(argv)
    if length(argv)<2
        error("analysis.jl program expects 2 or more arguments \
               \n    1. config file name.\
               \n    2... one or more strings of the form datasetname=filename where\
                        datasetname is a name to be used in plotting and filename is a\
                        the name of a file containing samples.")
    end
    ymax = parse(Float64, argv[end-1])
    islegend = argv[end] == "true"
    config = YAML.load_file(joinpath(pwd(), argv[1]))

    dataset_metainfo = split.(argv[2:end-2], '=')
    dataset_names = [info[1] for info in dataset_metainfo] 
    dataset_filenames = [join(info[2:end], "=") for info in dataset_metainfo] 
    
    nburnin = config["inference"]["mh_config"]["nadapt"]
    paramnames = [:R_0]

    chains = read_datasets(dataset_filenames, dataset_names, nburnin, paramnames)
    
    traceplots = maketraceplots(chains)

    densityplots = OrderedDict{Any, Any}()

    ctspriordists, discpriordists = makepriordists(config)
    priors = [ctspriordists...]
    paramid = 1
    for param in paramnames
        prior = priors[paramid]
        densityplots[param] = make1dposteriorpdf(chains, param, prior, ymax, islegend)
        paramid += 1
    end

    caseidentifier = "config_$(argv[1])_$(join(keys(chains), "-"))"
    caseidentifier = replace(caseidentifier, 
        "." => "_", " " => "_", "/" => "_", "\\" => "_")
    for (name, plt) in traceplots
        figfilename = joinpath(pwd(), "figs", "traceplot_$(name)_$(caseidentifier).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end

    for (name, plt) in densityplots
        figfilename = joinpath(pwd(), "figs", "density_$(name)_$(caseidentifier).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end

    return 
end

