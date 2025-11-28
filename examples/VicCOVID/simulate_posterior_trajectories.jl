using MultitypeBranchingProcessInference
using MCMCChains
using StatsPlots
using YAML
using LaTeXStrings
using KernelDensity
using Distributions
using Random 

include("./utils/config.jl")
include("./utils/figs.jl")
include("./utils/io.jl")

function read_datasets(filenames, datasetnames, nburnin)
    datasets = Dict{String, Array{Float64,2}}()
    fileid = 1
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
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

function main(argv)
    if length(argv)!=2
        error("analysis.jl program expects 2 or more arguments \
               \n    1. config file name.\
               \n    2. a string of the form datasetname=filename where\
                        datasetname is a name to be used in plotting and filename is a\
                        the name of a file containing samples.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    datafilearg = argv[2]
    datafilename, datafilepath = split(datafilearg, "=")

    samples = read_datasets([datafilepath], [datafilename], config["inference"]["mh_config"]["nadapt"])

    paramnames = [
        :R_0_1, :R_0_2, :R_0_3, :R_0_4, :R_0_5, :R_0_6, :R_0_7, :R_0_8, :R_0_9, :R_0_10, :R_0_11, :R_0_12, :R_0_13, :R_0_14, :Z0_E, :Z0_I, :LL
    ]

    chain = only(datasetstochains(samples, paramnames)).second;

    model, param_seq = makemodel(config)

    Ntrajectories = 4000
    sample_idx = rand(axes(chain.value.data,1), Ntrajectories)

    seirconfig = config["model"]["stateprocess"]["params"]

    seirconfig = config["model"]["stateprocess"]["params"]
    llparam_map! = (mtbpparams, R0, i) -> begin
        return param_map!(
            mtbpparams, 
            seirconfig["E_state_count"], 
            seirconfig["I_state_count"], 
            (R0, seirconfig["T_E"][i], seirconfig["T_I"][i]), 
            seirconfig["immigration_rate"]=="nothing" ? nothing : seirconfig["immigration_rate"], 
        )
    end
    nstates = seirconfig["E_state_count"] + seirconfig["I_state_count"] + 1 # add one for obs state
    
    nsteps = 7*14+1
    println("[INFO] Simulating...")

    c = 0
    for j in sample_idx
        c += 1
        pars = chain.value.data[j,1:end-1]

        for i in eachindex(param_seq.seq)
            llparam_map!(param_seq[i], pars[i], i)
        end
        # discretise the intial state
        i = length(param_seq.seq)
        z0 = round.(eltype(model.stateprocess.initial_state.events[1]), pars[(i+1):end])
        if any(z0 .< 0) || all(z0 .== 0)
            return -Inf
        end
        # set initial state of model
        initial_state = only(model.stateprocess.initial_state.events)
        initial_state[1:(nstates-1)] .= z0
        # set moments of intial state
        MultitypeBranchingProcesses.firstmoment!(
            model.stateprocess.initial_state.first_moments, 
            model.stateprocess.initial_state.events,
            model.stateprocess.initial_state.distribution
        )
        MultitypeBranchingProcesses.secondmoment!(
            model.stateprocess.initial_state.second_moments,
            model.stateprocess.initial_state.events,
            model.stateprocess.initial_state.distribution
        )

        rng = makerng(1234)

        mtbp = model.stateprocess

        init!(rng, mtbp)
        
        nextparamidx = firstindex(param_seq)
        nextparamtime = gettime(first(param_seq))
        iszero(nextparamtime) || error("First param timestamp in paramseq must be 0, got $nextparamtime")
        
        writefilename = "data/trajectories/trajectories_$(c).bin"

        open(joinpath(pwd(), writefilename), "w") do io
            tstep = 1
            t = zero(tstep)
            write(io, Int64(nsteps+1))
            writeparticle(io, mtbp.state, t)
            for _ in 1:nsteps    
                if t == nextparamtime
                    params, nextparamidx = iterate(param_seq, nextparamidx)
                    setparams!(model, params)
                    nextparamtime = (nextparamidx > length(param_seq)) ? Inf : gettime(param_seq[nextparamidx])
                elseif t > nextparamtime 
                    error("Parameters at timestamp $nextparamtime. Parameter timestamp must equal an observation timestamp.")
                end
                
                t += tstep
                simulate!(rng, mtbp, tstep)
                writeparticle(io, mtbp.state, t)
            end
        end
    end
    paths = []
    t = nothing 
    for ci in 1:c
        writefilename = "data/trajectories/trajectories_$(ci).bin"
        path, t = readparticles(joinpath(pwd(), writefilename))
        push!(paths, path)
    end

    cases = [paths[i][ti+1][3] for ti in t, i in 1:c]
    new_cases = diff(cases, dims=1)

    im = zeros(maximum(new_cases)+1, size(new_cases, 1))
    qs = zeros(2, size(new_cases, 1))
    for ti in axes(new_cases, 1)
        bw = mean(new_cases[ti,:])/4
        cases_kde = kde(new_cases[ti,:], bandwidth=bw)
        im[:, ti] .= pdf(cases_kde, 0:maximum(new_cases))
        im[:, ti] ./= maximum(im[:, ti])
        # im[:, ti] ./= sum(im[:, ti])
        # qs[:, ti] = quantile(im[:, ti], [0.8, 0.9])
    end
    # im[im .<= 0.1] .= -Inf
    # im[im .> qs[[2],:]] .= 1.0
    # im[qs[[2],:] .>= im .> qs[[1],:]] .= 0.5

    # plot(new_cases, color=:grey, alpha=0.1, label=false)
    raw_observations = read_observations(joinpath(pwd(), "data/observations.csv"))
    obs = [only(raw_observations[i]) for i in eachindex(raw_observations)]
    # plot!(obs)
    
    idx = (im .> 0) .* (1:size(im,1))
    heatmap(im, colorbar=true, ylims=(0,2000), grid=false, size=(600,400))
    plot!(repeat(0:length(obs)+1, inner=2)[2:end-3], repeat(obs, inner=2), color=:black, 
        # title="p=$(config["model"]["stateprocess"]["params"]["observation_probability"]), T_I=$(config["model"]["stateprocess"]["params"]["T_I"][1])"
        label="Observed cases", legend=:topleft
    )
    plot!(ylabel="Observed New Daily Cases")
    plot!(xlabel="Days")
    
    # plot()
    # for t in 7:7:98
    #     violin!([t]./14, new_cases[t,:], label=false, color=:red, alpha=0.5, side=:right)
    # end
    # plot!(twinx(), repeat(0:length(obs)+1, inner=2)[2:end-3]./14, repeat(obs, inner=2), color=:black, 
    #     # title="p=$(config["model"]["stateprocess"]["params"]["observation_probability"]), T_I=$(config["model"]["stateprocess"]["params"]["T_I"][1])"
    #     label="Observed cases"
    # )
    # plot!(ylabel="Cases")
    # plot!(xlabel="Days")
    # plot!(ylims=(0,2000))

    # quantile_series = zeros(size(new_cases, 1), 5)
    # for t in axes(new_cases, 1)
    #     quantile_series[t,:] = quantile(new_cases[t,:], [0.1;0.25;0.5;0.75;0.9])
    # end
    # plot(1:size(new_cases, 1)-1, quantile_series[1:end-1,3], ribbon=(quantile_series[1:end-1,3]-quantile_series[1:end-1,1],quantile_series[1:end-1,end]-quantile_series[1:end-1,3]), legend=false, color=:grey, label="90% CI")
    # plot!(1:size(new_cases, 1)-1, quantile_series[1:end-1,3], ribbon=(quantile_series[1:end-1,3]-quantile_series[1:end-1,2],quantile_series[1:end-1,end-1]-quantile_series[1:end-1,3]), legend=false, color=:grey, label="50% CI")
    # plot!(1:size(new_cases, 1)-1, quantile_series[1:end-1,3], color=:black)
    # plot!(1:size(new_cases, 1)-1, obs, color=:red)
    # display(plot!())
    
    savefig(plot!(), "figs/trajectories_p=$(config["model"]["stateprocess"]["params"]["observation_probability"]), T_I=$(config["model"]["stateprocess"]["params"]["T_I"][1]).svg")

    return plot!()
end

argv = ["config.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples.f64_array.bin"]
main(argv)
# argv = ["config_p_0.8_TI_3.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples_p_0.8_TI_3.f64_array.bin"]
# main(argv)
# argv = ["config_p_1_TI_3.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples_p_1_TI_3.f64_array.bin"]
# main(argv)
# argv = ["config_p_1_TI_8.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples_p_1_TI_8.f64_array.bin"]
# main(argv)
# argv = ["config_p_0.1_TI_3.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples_p_0.1_TI_3.f64_array.bin"]
# main(argv)
# argv = ["config_p_0.1_TI_8.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples_p_0.1_TI_8.f64_array.bin"]
# main(argv)
# argv = ["config_p_0.8_TI_8.yaml" "VicCOVID=data/expcovfn_vic_covid_kalman_param_samples_p_0.8_TI_8.f64_array.bin"]
# main(argv)
