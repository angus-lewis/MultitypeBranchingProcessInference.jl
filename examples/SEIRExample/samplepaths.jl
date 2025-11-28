using YAML
using StatsPlots
using MCMCChains
using OrderedCollections
using MultitypeBranchingProcessInference
using MultitypeBranchingProcessInference.MetropolisHastings
using Distributions

include("./utils/io.jl")
include("./utils/figs.jl")

function read_datasets(filenames, datasetnames, nburnin, paramnames, dtype=Float64)
    fileid = 1
    chains = OrderedDict{String, Chains}()
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io, dtype)
        end
        dataset = samples[1:end, (nburnin+1):end]
        print(size(dataset))
        chain = Chains(dataset', paramnames)
        open(joinpath(pwd(), "$(filename).summary.txt"), "w") do io
            display(TextDisplay(io), chain)
        end
        chains[datasetname] = chain
        fileid += 1
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

function generate_kf_posterior_sims(chain)
    sim_chain_data = zeros(length(chain),3*27)
    for i in 1:3
        for t in 0:26
            sim_chain_data[:,i+t*3] .= simulate_filtering_posterior(chain, i, t)
        end
    end
    names_ = vcat([Symbol.(["z_1", "z_2", "z_3"].*"_$(t)") for t in 0:26]...)
    chain = Chains(sim_chain_data, names_)
    return chain
end

nburnin = 20480

# SEIR
lowlowpath, t = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.12_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
lowpath, t = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.2_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
medpath, t100 = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
highpath, t = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.5_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
ts = [t, t100, t, t]

exposed = Vector{Vector{Int64}}(undef, 4)
exposed[1] = [lowlowpath[i][1] for i in 1:length(t)]
exposed[2] = [lowpath[i][1] for i in 1:length(t)]
exposed[3] = [medpath[i][1] for i in 1:length(t100)]
exposed[4] = [highpath[i][1] for i in 1:length(t)]

infectious = Vector{Vector{Int64}}(undef, 4)
infectious[1] = [lowlowpath[i][2] for i in 1:length(t)]
infectious[2] = [lowpath[i][2] for i in 1:length(t)]
infectious[3] = [medpath[i][2] for i in 1:length(t100)]
infectious[4] = [highpath[i][2] for i in 1:length(t)]

totals30 = [sum(exposed[i][1:6]+infectious[i][1:6]) for i in 1:4]
totals60 = [sum(exposed[i][1:16]+infectious[i][1:16]) for i in 1:4]
totals90 = [sum(exposed[i][1:26]+infectious[i][1:26]) for i in 1:4]

new_infectious = Vector{Vector{Int64}}(undef, 4)
new_infectious[1] = [lowlowpath[i][3] for i in 1:length(t)]
new_infectious[2] = [lowpath[i][3] for i in 1:length(t)]
new_infectious[3] = [medpath[i][3] for i in 1:length(t100)]
new_infectious[4] = [highpath[i][3] for i in 1:length(t)]

fillalpha = 0.35

let
    state_param_chains_kf = read_datasets(
        "./data/".*[
            "model_infose1i1r_infection_rate=0.12_nsteps=25_cov=[1.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.2_nsteps=25_cov=[1.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.5_nsteps=25_cov=[1.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin"
        ],
        ["R=0.12", "R=0.2", "R=0.3", "R=0.5"],
        nburnin,
        vcat([Symbol.(["z_1", "z_2", "z_3", "cov_11", "cov_12", "cov_13", "cov_21", "cov_22", "cov_23", "cov_31", "cov_32", "cov_33"].*"_$(t)") for t in 0:26]...)
    )
    state_chains_pf = read_datasets(
        "./data/".*[
            "model_infose1i1r_infection_rate=0.12_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.2_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.5_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.f64_array.bin"
        ],
        ["R=0.12", "R=0.2", "R=0.3", "R=0.5"],
        nburnin,
        vcat([Symbol.(["z_1", "z_2", "z_3"].*"_$(t)") for t in 0:26]...),
        Int
    )

    state_chains_kf = Dict(key => generate_kf_posterior_sims(state_param_chains_kf[key]) for key in keys(state_param_chains_kf))

    log_quantiles_kf = Dict(key => [log2.(1 .+ quantile(state_chains_kf[key][Symbol("z_$(i)_$(t)")].data[:,1], q)) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_kf))
    log_quantiles_pf = Dict(key => [log2.(1 .+ quantile(state_chains_pf[key][Symbol("z_$(i)_$(t)")].data[:,1], q)) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_pf))

    quantiles_kf = Dict(key => [quantile(state_chains_kf[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_kf))
    quantiles_pf = Dict(key => [quantile(state_chains_pf[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_pf))

    let
        p1 = plot(size=(600,400)); p2 = plot(size=(600,400)); p3 = plot(size=(600,400))
        colours = [_pastel_blue, _gray, _pastel_green, _pastel_red]
        labels = ["R0 = $(round(0.12/(3/28), digits=2))" "R0 = $(round(0.2/(3/28), digits=2))" "R0 = $(round(0.3/(3/28), digits=2))" "R0 = $(round(0.5/(3/28), digits=2))"]
        Rs = [0.12; 0.2; 0.3; 0.5]
        for i in [1;3;4]
            p1 = plot!(p1, ts[i], log2.(1 .+exposed[i]); ylabel="Exposed", legend=:topleft,
                label=labels[i],
                color=colours[i], linewidth=2)
            p1 = plot!(p1, ts[i], log_quantiles_kf["R=$(Rs[i])"][2:end,3,1]; ylabel="Exposed", legend=:topleft,
                label=false,
                color=colours[i], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(log_quantiles_kf["R=$(Rs[i])"][2:end,3,1] - log_quantiles_kf["R=$(Rs[i])"][2:end,1,1], log_quantiles_kf["R=$(Rs[i])"][2:end,5,1]-log_quantiles_kf["R=$(Rs[i])"][2:end,3,1]))
            p2 = plot!(p2, ts[i], log2.(1 .+infectious[i]); ylabel="Infectious", legend=false, color=colours[i], linewidth=2)
            p2 = plot!(p2, ts[i], log_quantiles_kf["R=$(Rs[i])"][2:end,3,2]; ylabel="Infectious", legend=false, color=colours[i], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(log_quantiles_kf["R=$(Rs[i])"][2:end,3,2] - log_quantiles_kf["R=$(Rs[i])"][2:end,1,2], log_quantiles_kf["R=$(Rs[i])"][2:end,5,2]-log_quantiles_kf["R=$(Rs[i])"][2:end,3,2]))
            p3 = plot!(p3, ts[i][2:end], log2.(1 .+diff(new_infectious[i], dims=1)); xlabel="Time (days)", ylabel="Observed", legend=false, color=colours[i], linewidth=2)
            p3 = plot!(p3, ts[i][1:end], log_quantiles_kf["R=$(Rs[i])"][2:end,3,3]; xlabel="Time (days)", ylabel="Observed", legend=false, color=colours[i], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(log_quantiles_kf["R=$(Rs[i])"][2:end,3,3] - log_quantiles_kf["R=$(Rs[i])"][2:end,1,3], log_quantiles_kf["R=$(Rs[i])"][2:end,5,3]-log_quantiles_kf["R=$(Rs[i])"][2:end,3,3]))
        end
        yl1 = ylims(p1)
        yl2 = ylims(p2)
        yl = (-0.4, max(yl1[2], yl2[2]))
        ylims!(p1, yl)
        ylims!(p2, yl)
        yt1 = yticks(p1)
        yt1[1][2] .= string.(Int.(2 .^yt1[1][1]) .-1)
        p1 = plot!(p1, ylim=yl, yticks=yt1[1])
        yt2 = yticks(p2)
        yt2[1][2] .= string.(Int.(2 .^yt2[1][1]) .-1)
        p2 = plot!(p2, ylim=yl, yticks=yt2[1])
        ylims!(p3, (-0.4, yl1[2]))
        yt3 = yticks(p3)
        yt3[1][2] .= string.(Int.(2 .^yt3[1][1]) .-1)
        p3 = plot!(p3, yticks=yt3[1])
        p = plot!(p1, p2, p3, layout=(3,1), size=(600,800), left_margin=4Plots.mm)

        savefig(p, "figs/seir_samplepaths.pdf")
    end

    let
        for i in [1;3;4]
            p1 = plot(size=(300,400)); p2 = plot(size=(300,400)); p3 = plot(size=(300,400))
            colours = [_pastel_blue, _pastel_red]
            labels1 = ["R0 = $(round(0.12/(3/28), digits=2))" "R0 = $(round(0.2/(3/28), digits=2))" "R0 = $(round(0.3/(3/28), digits=2))" "R0 = $(round(0.5/(3/28), digits=2))"]
            Rs = [0.12; 0.2; 0.3; 0.5]
            p1 = plot!(p1, ts[i], exposed[i]; ylabel="Exposed", legend=:topleft,
                label=false,
                color=_gray, linewidth=2)
            p2 = plot!(p2, ts[i], infectious[i]; ylabel="Infectious", legend=false, color=_gray, linewidth=2)
            p3 = plot!(p3, ts[i][2:end], diff(new_infectious[i], dims=1); xlabel="Time (days)", ylabel="Observed", legend=false, color=_gray, linewidth=2)
            c = 1
            labels = ["Gaussian" "Particle"]
            for quantiles in (quantiles_kf, quantiles_pf)
                p1 = plot!(p1, ts[i], quantiles["R=$(Rs[i])"][2:end,3,1]; ylabel="Exposed", legend=:topleft,
                    label=labels[c],
                    color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                    ribbon=(quantiles["R=$(Rs[i])"][2:end,3,1] - quantiles["R=$(Rs[i])"][2:end,1,1], quantiles["R=$(Rs[i])"][2:end,5,1]-quantiles["R=$(Rs[i])"][2:end,3,1]))
                p2 = plot!(p2, ts[i], quantiles["R=$(Rs[i])"][2:end,3,2]; ylabel="Infectious", legend=false, color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                    ribbon=(quantiles["R=$(Rs[i])"][2:end,3,2] - quantiles["R=$(Rs[i])"][2:end,1,2], quantiles["R=$(Rs[i])"][2:end,5,2]-quantiles["R=$(Rs[i])"][2:end,3,2]))
                p3 = plot!(p3, ts[i][1:end], quantiles["R=$(Rs[i])"][2:end,3,3]; xlabel="Time (days)", ylabel="Observed", legend=false, color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                    ribbon=(quantiles["R=$(Rs[i])"][2:end,3,3] - quantiles["R=$(Rs[i])"][2:end,1,3], quantiles["R=$(Rs[i])"][2:end,5,3]-quantiles["R=$(Rs[i])"][2:end,3,3]))
                c += 1
            end
            yl1 = ylims(p1)
            # yl2 = ylims(p2)
            yl = (yl1[1], yl1[2]+2)
            ylims!(p1, yl)
            # ylims!(p2, yl)
            # yt1 = yticks(p1)
            # yt1[1][2] .= string.(Int.(2 .^yt1[1][1]) .-1)
            # p1 = plot!(p1, ylim=yl, yticks=yt1[1])
            # yt2 = yticks(p2)
            # yt2[1][2] .= string.(Int.(2 .^yt2[1][1]) .-1)
            # p2 = plot!(p2, ylim=yl, yticks=yt2[1])
            # ylims!(p3, (-0.4, yl1[2]))
            # yt3 = yticks(p3)
            # yt3locs = yt3[1][1][1]:yt3[1][1][end]
            # yt3 = [(yt3locs, string.(Int.(2 .^yt3locs) .-1))]
            # p3 = plot!(p3, yticks=yt3[1])
            p = plot!(p1, p2, p3, layout=(3,1), size=(300,800), left_margin=7Plots.mm)
            
            savefig(p, "figs/seir_filtering_posteriors_Rexperiment_$(labels1[i]).pdf")
        end
    end
end

let
    state_param_chains_kf = read_datasets(
        "./data/".*[
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.5_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=1.0_method=kalman_filter.f64_array.bin"
        ],
        ["p=0.5", "p=0.75", "p=1"],
        nburnin,
        vcat([Symbol.(["z_1", "z_2", "z_3", "cov_11", "cov_12", "cov_13", "cov_21", "cov_22", "cov_23", "cov_31", "cov_32", "cov_33"].*"_$(t)") for t in 0:26]...)
    )
    state_chains_pf = read_datasets(
        "./data/".*[
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.5_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=1.0_method=particle_filter.f64_array.bin"
        ],
        ["p=0.5", "p=0.75", "p=1"],
        nburnin,
        vcat([Symbol.(["z_1", "z_2", "z_3"].*"_$(t)") for t in 0:26]...),
        Int
    )

    state_chains_kf = Dict(key => generate_kf_posterior_sims(state_param_chains_kf[key]) for key in keys(state_param_chains_kf))

    quantiles_kf = Dict(key => [quantile(state_chains_kf[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_kf))
    quantiles_pf = Dict(key => [quantile(state_chains_pf[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_pf))

    for i in [1;2;3]
        p1 = plot(size=(300,400)); p2 = plot(size=(300,400)); p3 = plot(size=(300,400))
        colours = [_pastel_blue, _pastel_red]
        labels1 = ["p = $(round(0.5, digits=2))" "p = $(round(0.75, digits=2))" "p = $(round(1, digits=2))"]
        Rs = Real[0.5, 0.75, 1]
        p1 = plot!(p1, ts[i], exposed[3]; ylabel="Exposed", legend=:topleft,
            label=false,
            color=_gray, linewidth=2)
        p2 = plot!(p2, ts[i], infectious[3]; ylabel="Infectious", legend=false, color=_gray, linewidth=2)
        p3 = plot!(p3, ts[i][2:end], diff(new_infectious[3], dims=1); xlabel="Time (days)", ylabel="Observed", legend=false, color=_gray, linewidth=2)
        c = 1
        labels = ["Gaussian" "Particle"]
        for quantiles in (quantiles_kf, quantiles_pf)
            p1 = plot!(p1, ts[i], quantiles["p=$(Rs[i])"][2:end,3,1]; ylabel="Exposed", legend=:topleft,
                label=labels[c],
                color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(quantiles["p=$(Rs[i])"][2:end,3,1] - quantiles["p=$(Rs[i])"][2:end,1,1], quantiles["p=$(Rs[i])"][2:end,5,1]-quantiles["p=$(Rs[i])"][2:end,3,1]))
            p2 = plot!(p2, ts[i], quantiles["p=$(Rs[i])"][2:end,3,2]; ylabel="Infectious", legend=false, color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(quantiles["p=$(Rs[i])"][2:end,3,2] - quantiles["p=$(Rs[i])"][2:end,1,2], quantiles["p=$(Rs[i])"][2:end,5,2]-quantiles["p=$(Rs[i])"][2:end,3,2]))
            p3 = plot!(p3, ts[i][1:end], quantiles["p=$(Rs[i])"][2:end,3,3]; xlabel="Time (days)", ylabel="Observed", legend=false, color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(quantiles["p=$(Rs[i])"][2:end,3,3] - quantiles["p=$(Rs[i])"][2:end,1,3], quantiles["p=$(Rs[i])"][2:end,5,3]-quantiles["p=$(Rs[i])"][2:end,3,3]))
            c += 1
        end
        # yl1 = ylims(p1)
        # yl2 = ylims(p2)
        # yl = (-0.4, max(yl1[2], yl2[2]))
        # ylims!(p1, yl)
        # ylims!(p2, yl)
        # yt1 = yticks(p1)
        # yt1[1][2] .= string.(Int.(2 .^yt1[1][1]) .-1)
        # p1 = plot!(p1, ylim=yl, yticks=yt1[1])
        # yt2 = yticks(p2)
        # yt2[1][2] .= string.(Int.(2 .^yt2[1][1]) .-1)
        # p2 = plot!(p2, ylim=yl, yticks=yt2[1])
        # ylims!(p3, (-0.4, yl1[2]))
        # yt3 = yticks(p3)
        # yt3locs = yt3[1][1][1]:yt3[1][1][end]
        # yt3 = [(yt3locs, string.(Int.(2 .^yt3locs) .-1))]
        # p3 = plot!(p3, yticks=yt3[1])
        p = plot!(p1, p2, p3, layout=(3,1), size=(300,800), left_margin=7Plots.mm)
        
        savefig(p, "figs/seir_filtering_posteriors_pexperiment_$(labels1[i]).pdf")
    end
end

let
    state_param_chains_kf = read_datasets(
        "./data/".*[
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[0.25]_observation_probability=0.75_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[4.0]_observation_probability=0.75_method=kalman_filter.f64_array.bin"
        ],
        ["var=0.25", "var=1", "var=4"],
        nburnin,
        vcat([Symbol.(["z_1", "z_2", "z_3", "cov_11", "cov_12", "cov_13", "cov_21", "cov_22", "cov_23", "cov_31", "cov_32", "cov_33"].*"_$(t)") for t in 0:26]...)
    )
    state_chains_pf = read_datasets(
        "./data/".*[
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[0.25]_observation_probability=0.75_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[1.0]_observation_probability=0.75_method=particle_filter.f64_array.bin",
            "model_infose1i1r_infection_rate=0.3_nsteps=25_cov=[4.0]_observation_probability=0.75_method=particle_filter.f64_array.bin"
        ],
        ["var=0.25", "var=1", "var=4"],
        nburnin,
        vcat([Symbol.(["z_1", "z_2", "z_3"].*"_$(t)") for t in 0:26]...),
        Int
    )

    state_chains_kf = Dict(key => generate_kf_posterior_sims(state_param_chains_kf[key]) for key in keys(state_param_chains_kf))

    quantiles_kf = Dict(key => [quantile(state_chains_kf[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_kf))
    quantiles_pf = Dict(key => [quantile(state_chains_pf[key][Symbol("z_$(i)_$(t)")].data[:,1], q) for t in 0:26, q in [0.1;0.25;0.5;0.75;0.9], i in 1:3] for key in keys(state_chains_pf))

    for i in [1;2;3]
        p1 = plot(size=(300,400)); p2 = plot(size=(300,400)); p3 = plot(size=(300,400))
        colours = [_pastel_blue, _pastel_red]
        labels1 = ["var = $(round(0.25, digits=2))" "var = $(round(1., digits=2))" "var = $(round(4., digits=2))"]
        Rs = Real[0.25, 1, 4]
        p1 = plot!(p1, ts[i], exposed[3]; ylabel="Exposed", legend=:topleft,
            label=false,
            color=_gray, linewidth=2)
        p2 = plot!(p2, ts[i], infectious[3]; ylabel="Infectious", legend=false, color=_gray, linewidth=2)
        p3 = plot!(p3, ts[i][2:end], diff(new_infectious[3], dims=1); xlabel="Time (days)", ylabel="Observed", legend=false, color=_gray, linewidth=2)
        c = 1
        labels = ["Gaussian" "Particle"]
        for quantiles in (quantiles_kf, quantiles_pf)
            p1 = plot!(p1, ts[i], quantiles["var=$(Rs[i])"][2:end,3,1]; ylabel="Exposed", legend=:topleft,
                label=labels[c],
                color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(quantiles["var=$(Rs[i])"][2:end,3,1] - quantiles["var=$(Rs[i])"][2:end,1,1], quantiles["var=$(Rs[i])"][2:end,5,1]-quantiles["var=$(Rs[i])"][2:end,3,1]))
            p2 = plot!(p2, ts[i], quantiles["var=$(Rs[i])"][2:end,3,2]; ylabel="Infectious", legend=false, color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(quantiles["var=$(Rs[i])"][2:end,3,2] - quantiles["var=$(Rs[i])"][2:end,1,2], quantiles["var=$(Rs[i])"][2:end,5,2]-quantiles["var=$(Rs[i])"][2:end,3,2]))
            p3 = plot!(p3, ts[i][1:end], quantiles["var=$(Rs[i])"][2:end,3,3]; xlabel="Time (days)", ylabel="Observed", legend=false, color=colours[c], linewidth=2, linestyle=:dot, fillalpha=fillalpha,
                ribbon=(quantiles["var=$(Rs[i])"][2:end,3,3] - quantiles["var=$(Rs[i])"][2:end,1,3], quantiles["var=$(Rs[i])"][2:end,5,3]-quantiles["var=$(Rs[i])"][2:end,3,3]))
            c += 1
        end
        # yl1 = ylims(p1)
        # yl2 = ylims(p2)
        # yl = (-0.4, max(yl1[2], yl2[2]))
        # ylims!(p1, yl)
        # ylims!(p2, yl)
        # yt1 = yticks(p1)
        # yt1[1][2] .= string.(Int.(2 .^yt1[1][1]) .-1)
        # p1 = plot!(p1, ylim=yl, yticks=yt1[1])
        # yt2 = yticks(p2)
        # yt2[1][2] .= string.(Int.(2 .^yt2[1][1]) .-1)
        # p2 = plot!(p2, ylim=yl, yticks=yt2[1])
        # ylims!(p3, (-0.4, yl1[2]))
        # yt3 = yticks(p3)
        # yt3locs = yt3[1][1][1]:yt3[1][1][end]
        # yt3 = [(yt3locs, string.(Int.(2 .^yt3locs) .-1))]
        # p3 = plot!(p3, yticks=yt3[1])
        p = plot!(p1, p2, p3, layout=(3,1), size=(300,800), left_margin=7Plots.mm)
        
        savefig(p, "figs/seir_filtering_posteriors_varexperiment_$(labels1[i]).pdf")
    end
end
