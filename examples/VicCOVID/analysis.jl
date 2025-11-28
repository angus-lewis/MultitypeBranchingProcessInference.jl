using MultitypeBranchingProcessInference
using MCMCChains
using StatsPlots
using YAML
using LaTeXStrings
using KernelDensity
using Distributions

include("./utils/config.jl")
include("./utils/figs.jl")

function read_datasets(filenames, datasetnames, nburnin, dtype=Float64)
    datasets = Dict{String, Array{dtype,2}}()
    fileid = 1
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io, dtype)
        end
        datasets[datasetname] = samples[:, (nburnin+1):end]
        fileid += 1
    end
    return datasets
end

function datasetstochains(datasets, paramnames)
    chains = Dict{String, Chains}()
    for (datasetname, dataset) in datasets
        chain = Chains(dataset', paramnames)
        chains[datasetname] = chain
    end
    return chains
end


function simulate_filtering_posterior(chain, state_ix, t)
    sims = zeros(length(chain))
    mus = chain[Symbol("z_$(state_ix)_$(t)")]
    covs = chain[Symbol("cov_$(state_ix)$(state_ix)_$(t)")]
    for i in 1:length(chain)
        Z = Normal(mus[i], sqrt(covs[i]))
        sims[i] = rand(Z)
    end
    return sims
end

function generate_kf_posterior_sims(chain, observations)
    sim_chain_data = zeros(length(chain),3*(length(observations)+1))
    for i in 1:3
        for t in 0:length(observations)
            sim_chain_data[:,i+t*3] .= simulate_filtering_posterior(chain, i, t)
        end
    end
    names_ = vcat([Symbol.(["z_1", "z_2", "z_3"].*"_$(t)") for t in 0:length(observations)]...)
    chain = Chains(sim_chain_data, names_)
    return chain
end

function main(argv)
    if length(argv)!=3
        error("analysis.jl program expects 3 arguments \
               \n    1. config file name.\
               \n    2. a string of the form datasetname=filename where\
                        datasetname is a name to be used in plotting and filename is a\
                        the name of a file containing samples.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    datafilearg = argv[2]
    datafilename, datafilepath = split(datafilearg, "=")

    model_info_file_arg = argv[3]
    modelinfofilename, modelinfofilepath = split(model_info_file_arg, "=")

    samples = read_datasets([datafilepath], [datafilename], config["inference"]["mh_config"]["nadapt"])

    observations = read_observations(joinpath(pwd(), config["inference"]["data"]["filename"]))
    observations = vcat(observations...)

    paramnames = [
        :R_0_1, :R_0_2, :R_0_3, :R_0_4, :R_0_5, :R_0_6, :R_0_7, :R_0_8, :R_0_9, :R_0_10, :R_0_11, :R_0_12, :R_0_13, :R_0_14, :Z0_E, :Z0_I, :LL
    ]

    chains_ = datasetstochains(samples, paramnames)
    summaryfilename = "$datafilepath-samples_file_$(argv[2])"
    summaryfilename = replace(summaryfilename, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    chainsummaryfilename = joinpath(pwd(), "data", summaryfilename)
    chainsummaryfilename = "$chainsummaryfilename.summary.txt"
    open(chainsummaryfilename, "w") do io
        display(TextDisplay(io), chains_[datafilename])
    end

    state_estimates_data = read_datasets([modelinfofilepath], [modelinfofilename], config["inference"]["mh_config"]["nadapt"]÷config["inference"]["mh_config"]["model_info_thin"])
    state_names = (
        vcat([Symbol.(["z_1", "z_2", "z_3", "cov_11", "cov_12", "cov_13", "cov_21", "cov_22", "cov_23", "cov_31", "cov_32", "cov_33"].*"_$(t)") for t in 0:length(observations)]...)
    )
    state_estimates = datasetstochains(state_estimates_data, state_names)
    statesummaryfilename = "$modelinfofilename-samples_file_$(argv[2])"
    statesummaryfilename = replace(statesummaryfilename, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    statechainsummaryfilename = joinpath(pwd(), "data", statesummaryfilename)
    statechainsummaryfilename = "$statechainsummaryfilename.summary.txt"
    open(statechainsummaryfilename, "w") do io
        display(TextDisplay(io), state_estimates[modelinfofilename])
    end

    state_samples = Dict(key => generate_kf_posterior_sims(state_estimates[key], observations) for key in keys(state_estimates))
    display(state_samples)
    quantiles = Dict(key => [quantile(state_samples[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:length(observations), q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_samples))

    # diagnostic trace plot
    trace_plt = plot(chains_[datafilename])

    trace_plt_figname = "traceplot-config_file_$(argv[1])-samples_file_$(argv[2])-dataset_$(config["inference"]["data"]["filename"])"
    trace_plt_figname = replace(trace_plt_figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    trace_plt_figname = joinpath(pwd(), "figs", "$trace_plt_figname.$FIGURE_FILE_EXT")
    savefig(trace_plt, trace_plt_figname)

    # param inference plot
    R0idx = [:R_0_1, :R_0_2, :R_0_3, :R_0_4, :R_0_5, :R_0_6, :R_0_7, :R_0_8, :R_0_9, :R_0_10, :R_0_11, :R_0_12, :R_0_13, :R_0_14]
    p = plot(size=(600,400))
    # kdes = []
    # max_kde = -Inf
    for i in eachindex(R0idx)
        violin!(p, [(i-1)*0.5], chains_[datafilename][R0idx[i]][:];
            ylims=(0, 2.5), color=:red, alpha=0.5, ylabel=L"R_0", side=:right,
            label=i==firstindex(R0idx) ? "Density" : false, legend=:topleft)
        # kdeR0 = kde(chains_[datafilename][R0idx[i]][:])
        # push!(kdes, [kdeR0.x, kdeR0.density])
        # max_kde = max(max_kde, maximum(kdeR0.density))
    end
    # for i in eachindex(R0idx)
    #     plot!(p, (i-1)*0.5 .+ kdes[i][2]./max_kde*0.5, kdes[i][1];
    #         ylims=(0, length(R0idx)/2-0.5), color=:black, label=false)
    #     plot!(p, (i-1)*0.5 .+ kdes[i][2]./max_kde*0.5, kdes[i][1];
    #         ylims=(0, length(R0idx)/2-0.5), fill=true, color=:red, alpha=0.5, ylabel=L"R_0", side=:right,
    #         label=i==firstindex(R0idx) ? "Density" : false, legend=:topleft)
    # end
    x = range(0, length(R0idx)/2, length=length(observations)+1)
    x = repeat(x, inner=2)
    x = x[2:end-1]
    y = repeat(observations, inner=2)
   
    q = twinx(p)
    plot!(q, x, y;
        color=:black, label="Daily cases", ylabel="Daily confirmed cases", legend=:topright)
    plot!(q, x, repeat(quantiles[modelinfofilename][2:end,3,3], inner=2);
        label="Estimate",
        color=_pastel_blue, linewidth=2, linestyle=:dot, fillalpha=0.35,
        ribbon=(
            repeat(quantiles[modelinfofilename][2:end,3,3] - quantiles[modelinfofilename][2:end,1,3], inner=2), 
            repeat(quantiles[modelinfofilename][2:end,5,3] - quantiles[modelinfofilename][2:end,3,3], inner=2))
        )
    ylims_ = ylims(p)
    for x in 0.5:0.5:(length(R0idx)/2-0.5)
        plot!(p, [x;x], [ylims_[1]; ylims_[2]]; color=:grey, linestyle=:dash, label=false)
    end
    plot!(p, xlims=(-0.5/7,7+0.5/7))
    plot!(p, xticks=(0:0.5:7, ["$(7*(i-1))" for i in 1:length(R0idx)+1]), xlabel="Days")
    plot!(p, grid=:off)

    p2 = plot(size=(600,400))
    plot!(p2, yticks=([0], [""]))
    q2 = twinx(p2)
    plot!(q2, x, repeat(quantiles[modelinfofilename][2:end,3,2], inner=2);
        label="Estimate",
        ylabel="Infectious", legend=:topright,
        color=_pastel_blue, linewidth=2, linestyle=:dot, fillalpha=0.35,
        ribbon=(
            repeat(quantiles[modelinfofilename][2:end,3,2] - quantiles[modelinfofilename][2:end,1,2], inner=2), 
            repeat(quantiles[modelinfofilename][2:end,5,2] - quantiles[modelinfofilename][2:end,3,2], inner=2))
        )
    ylims_ = ylims(p2)
    for x in 0.5:0.5:(length(R0idx)/2-0.5)
        plot!(p2, [x;x], [ylims_[1]; ylims_[2]]; color=:grey, linestyle=:dash, label=false)
    end
    plot!(p2, xlims=(-0.5/7,7+0.5/7))
    plot!(p2, xticks=(0:0.5:7, ["$(7*(i-1))" for i in 1:length(R0idx)+1]))
    plot!(p2, grid=:off)
    # plot!(p2, xlabel=nothing, ylabel=nothing)

    p1 = plot(size=(600,400))
    plot!(p1, yticks=([0], [""]))
    q1 = twinx(p1)
    plot!(q1, x, repeat(quantiles[modelinfofilename][2:end,3,1], inner=2);
        label="Estimate",
        ylabel="Exposed", legend=:topright,
        color=_pastel_blue, linewidth=2, linestyle=:dot, fillalpha=0.35,
        ribbon=(
            repeat(quantiles[modelinfofilename][2:end,3,1] - quantiles[modelinfofilename][2:end,1,1], inner=2), 
            repeat(quantiles[modelinfofilename][2:end,5,1] - quantiles[modelinfofilename][2:end,3,1], inner=2))
        )
    ylims_ = ylims(p1)
    for x in 0.5:0.5:(length(R0idx)/2-0.5)
        plot!(p1, [x;x], [ylims_[1]; ylims_[2]]; color=:grey, linestyle=:dash, label=false)
    end
    plot!(p1, xlims=(-0.5/7,7+0.5/7))
    plot!(p1, xticks=(0:0.5:7, ["$(7*(i-1))" for i in 1:length(R0idx)+1]))
    plot!(p1, grid=:off)
    # plot!(p1, xlabel=nothing, ylabel=nothing)

    plt = plot!(p1, p2, p, layout=(3,1), size=(600,800), left_margin=4Plots.mm, right_margin=4Plots.mm)
    
    figname = "R_0_densities_and_cases-config_file_$(argv[1])-samples_file_$(argv[2])-dataset_$(config["inference"]["data"]["filename"])"
    figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")
    savefig(plt, figname)

    return plt
end
args = [
    "config.yaml";
    "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples.f64_array.bin";
    "VicCOVID=data/model_info_expcovfn_vic_covid_kalman.info.txt"
]
p = main(args)
# main(ARGS)