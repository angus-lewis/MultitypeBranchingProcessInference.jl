using StatsPlots
using MCMCChains
using YAML
using MultitypeBranchingProcessInference
using DelimitedFiles
using Random

include("src/forecasting.jl")

function read_datasets(filenames, datasetnames, nburnin, paramnames)
    fileid = 1
    chains = Dict{String, Union{Chains,Array}}()
    for filename in filenames
        datasetname = datasetnames[fileid]
        if occursin(".bin", filename)
            if occursin("param", filename)
                names = paramnames[1]
            else
                names = paramnames[2]
            end
            samples = open(joinpath(pwd(), filename), "r") do io
                MetropolisHastings.read_binary_array_file(io)
            end
            dataset = Matrix(samples[:, (nburnin+1):end]')
            @show names
            @show filename
            chain = Chains(dataset, names)
            open(joinpath(pwd(), "$(filename).summary.txt"), "w") do io
                display(TextDisplay(io), chain)
            end
            chains[datasetname] = chain
        elseif occursin(".csv", filename)
            dataset, h = readdlm(filename, ','; header=true)
            chains[datasetname] = dataset
        end
        
        fileid += 1
    end
    return chains
end

function evalmeanfun(samps, times)
    out = zeros(length(times), size(samps,2))
    for i in axes(samps,2)
        fun = Logistic4(samps[1:4,i]...)
        out[:,i] .= fun.(times)
    end
    return out
end

function main(argv)
    if length(argv)<2
        error("analysis.jl program expects 2 or more arguments \
               \n    1. config file name.\
               \n    2... one or more strings of the form datasetname=filename where\
                        datasetname is a name to be used in plottinr and filename is a\
                        the name of a file containing samples.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))
    
    dataset_metainfo = split.(argv[2:end], '=')
    dataset_names = [info[1] for info in dataset_metainfo] 
    dataset_filenames = [info[2] for info in dataset_metainfo] 

    nburnin = config["inference"]["mh_config"]["nadapt"]
    paramnames = Symbol.(config["model"]["inferred_parameters"]["param_names"])
    infonames = Symbol.(config["model"]["info_param_names"])

    chains = read_datasets(dataset_filenames, dataset_names, nburnin, (paramnames, infonames))

    param_chains = [chains[i] for i in keys(chains) if occursin("param", i)]
    state_chains = [chains[i] for i in keys(chains) if occursin("state", i)]
    forecasts = [chains[i] for i in keys(chains) if occursin("forecast", i)]
    
    q1 = plot(param_chains[first(keys(param_chains))])
    for i in Iterators.drop(keys(param_chains), 1)
        plot!(q1, param_chains[i])
    end
    plot!(q1)

    mkpath(joinpath(pwd(), "figs"))
    fname = replace(argv[1]*"_"*argv[2]*"_traceplots_params.png", "/"=>"_")
    savefig(q1, joinpath(pwd(), "figs", fname))

    q2 = plot(state_chains[first(keys(state_chains))])
    for i in Iterators.drop(keys(state_chains), 1)
        plot!(q2, state_chains[i])
    end
    plot!(q2)

    mkpath(joinpath(pwd(), "figs"))
    fname = replace(argv[1]*"_"*argv[2]*"_traceplots_state.png", "/"=>"_")
    savefig(q2, joinpath(pwd(), "figs", fname))

    observations = read_observations(joinpath(pwd(), config["data"]["filename"]))
    observations = vcat(observations...)

    dow_prior = config["model"]["fixed_parameters"]["observation_model"]["day_of_week_effect_prior"]
    @show dow_effect = estimate_dow_effect(observations, 0, dow_prior, length(dow_prior))

    p = plot()
    xi = 0
    timestamps = vcat(
        config["model"]["inferred_parameters"]["R_0"]["changepoints"],
        [length(observations)]
    )
    dt = diff(timestamps)
    maxtstep = maximum(dt)
    for datasetname in keys(param_chains)
        dataset = param_chains[datasetname]
        i = 0
        for name in paramnames
            if !occursin("R", string(name))
                continue
            end
            i += 1
            violin!(p, [xi], dataset[name][:];
                color=:red, alpha=0.2, ylabel="R_0", side=:right,
                label=i==1 ? "Density" : false, legend=:topleft)
            xi += 0.5 * dt[i]/maxtstep   
        end
    end
    xr = range(0, xi; length=length(observations))
    x = repeat(xr, inner=2)
    x = x[2:end]
    push!(x, last(x)+step(xr))
    y = repeat(observations, inner=2)
    sp = twinx(p)
    plot!(sp, x, y;
        color=:black, label="Daily cases", ylabel="Daily confirmed cases", legend=:topright)
    yl = ylims(sp)
    
    stateoffset = (
        config["model"]["fixed_parameters"]["E_state_count"]
        + config["model"]["fixed_parameters"]["I_state_count"]
        + 1
        + (config["model"]["fixed_parameters"]["notification_rate"] != "nothing")
    )
    nstates = stateoffset = (
        config["model"]["fixed_parameters"]["E_state_count"]
        + config["model"]["fixed_parameters"]["I_state_count"]
        + 1
        + (config["model"]["fixed_parameters"]["notification_rate"] != "nothing")
        + (config["model"]["fixed_parameters"]["immigration_rate"] != "nothing")
    )
    i = 0
    rng = Random.default_rng(1234)
    @show last_obs_dow_effect = dow_effect[(length(observations)-1)%length(dow_effect) + 1]
    for datasetname in keys(state_chains)
        i += 1
        dataset = state_chains[datasetname]
        mus = dataset[infonames[stateoffset]][:] .* last_obs_dow_effect
        vars = dataset[infonames[nstates + nstates*(stateoffset-1)+stateoffset]][:] .* last_obs_dow_effect^2
        z = [rand(rng, Normal(mus[i], vars[i])) for i in eachindex(mus)]
        violin!(sp, [xi], z;
            color=:blue, alpha=0.2, ylabel="", side=:right,
            label=i==1 ? "State" : false, legend=:topleft)
    end

    forecasttimes = config["forecast"]["times"]
    xf = [forecasttimes[i]*step(xr) for i in eachindex(forecasttimes)]
    for datasetname in keys(forecasts)
        dataset = forecasts[datasetname]
        plot!(sp, xf, dataset[:, 2:end];
            color=:purple, alpha=0.2, ylabel="", side=:right,
            label=false, legend=:topleft, ylims=(yl[1], yl[2]*2))
    end

    fname = replace(argv[1]*"_"*argv[2]*"_posteriors.png", "/"=>"_")
    savefig(p, joinpath(pwd(), "figs", fname))

    (r0paths, ~) = readdlm(config["forecast"]["outfilename"]["r0paths"], ','; header=true)
    r = plot(r0paths[:,1], r0paths[:,2:end]; color=:blue, alpha=0.1, label=false)
    fname = replace(argv[1]*"_"*argv[2]*"_r0path.png", "/"=>"_")
    savefig(r, joinpath(pwd(), "figs", fname))
    return q1, q2, p
end

argv = [
    "config.yaml"; 
    "VicCovid_param=data/dow_expcovfn_vic_covid_kalman_param_samples.f64_array.bin";
    "VicCovid_state=data/dow_expcovfn_vic_covid_kalman_state.model_info.f64_array.bin";
    "VicCovid_forecast=data/dow_forecasts.csv";
]

q1, q2, p = main(argv);