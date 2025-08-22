using StatsPlots
using YAML
using Distributions
using LaTeXStrings
using MultitypeBranchingProcessInference

default(; fontfamily="Bookman")

function readparticles(fn)
    particles = open(fn, "r") do io 
        nsteps = read(io, Int64)
        particles = Dict{Int64, Matrix{Int64}}()
        for step in 1:nsteps
            tstamp = read(io, Int64)
            n_particles = read(io, Int64)
            particle_length = read(io, Int64)
            step_particles = Matrix{Int64}(undef, n_particles, particle_length)
            for particleidx in 1:n_particles
                for eltidx in 1:particle_length
                    elt = read(io, Int64)
                    step_particles[particleidx, eltidx] = elt
                end
            end
            particles[tstamp] = step_particles
        end
        particles
    end
    return particles
end

params = YAML.load_file(joinpath(pwd(), ARGS[1]))

if (params["E_immigration_rate"] == zero(params["E_immigration_rate"]) 
    && params["I_immigration_rate"] == zero(params["I_immigration_rate"]))

    immigration = nothing
else
    immigration = [params["E_immigration_rate"], params["I_immigration_rate"]]
end

mtbp = StateSpaceModels.SEIR(
    1, 1,
    params["infection_rate"], params["exposed_stage_chage_rate"], params["infectious_stage_chage_rate"], 
    params["observation_probability"], 
    nothing, # notification rate
    immigration,
    [params["initial_E"], params["initial_I"], params["initial_O"]],
)

particles = readparticles(joinpath(pwd(), ARGS[2]))

function makeplot(particles, mtbp)
    moments = MTBPMomentsOperator(mtbp)
    init!(mtbp)

    series = zeros(eltype(particles[0]), length(particles), mtbp.ntypes+1)
    sorted_keys = sort(collect(keys(particles)))
    plot()
    plot!([NaN], [NaN]; color=:grey, alpha=0.5, label = "Sample paths")
    for sampleidx in 1:size(particles[0], 1)
        count = 0
        for t in sorted_keys
            count += 1
            series[count, 1] = t
            series[count, 2:end] = particles[t][sampleidx,:]
        end
        plot!(series[:,2], series[:,3], alpha=0.1, color=:grey, label=false)
    end
    plot!(xlabel="Exposed", ylabel="Infectious")
    
    scattertimes = [20; 60; 100]
    pastel_blue = RGB(0.7, 0.8, 1.0)
    pastel_green = RGB(0.7, 1.0, 0.7)
    pastel_red = RGB(1.0, 0.7, 0.7)
    colours = [pastel_blue, pastel_green, pastel_red]
    i = 1
    for t in scattertimes
        scatter!(particles[t][:,1], particles[t][:,2], color=colours[i], label="t=$t", markersize=3)
        i += 1
    end
    i = 1
    for t in scattertimes
        moments!(moments, mtbp, t)
        mu = mean(moments, mtbp.state)
        sigma = variance_covariance(moments, mtbp.state)
        # covellipse!(mu[1:2], sigma[1:2,1:2]; n_std = 1, color=:lightblue, label=nothing, alpha=0.4)
        covellipse!(mu[1:2], sigma[1:2,1:2]; n_std = 2, color=colours[i], label=nothing, alpha=0.4)
        i += 1
    end

    mu_series = zeros(length(particles), mtbp.ntypes+1)
    count = 0
    for t in sorted_keys
        count += 1
        moments!(moments, mtbp, t)
        mu = zeros(paramtype(mtbp), getntypes(mtbp))
        mean!(mu, moments, mtbp.state)
        mu_series[count,1] = t
        mu_series[count, 2:end] = mu
        if t in scattertimes
            scatter!([mu_series[count, 2]], [mu_series[count, 3]], markersize=10, markershape=:+, color=:black, label=nothing)
            # annotate!([mu_series[count, 2]], [mu_series[count, 3]], "t=$t")
        end
    end
    plot!(mu_series[:,2], mu_series[:,3], color=:black, label=L"E[z(t)]", linewidth=2)
    
    return plot!(grid=nothing)
end
samplepathplot = makeplot(particles, mtbp)
savefig(samplepathplot, joinpath("figs", "SEIRToyModelSamplePathPlot.pdf"))

function makeqq(particles, mtbp)
    moments = MTBPMomentsOperator(mtbp)
    init!(mtbp)
    plots = Plots.Plot[]
    for t in [20, 60, 100]
        StateSpaceModels.moments!(moments, mtbp, t)
        mu = zeros(paramtype(mtbp), getntypes(mtbp))
        mean!(mu, moments, mtbp.state)
        vcov = zeros(paramtype(mtbp), getntypes(mtbp), getntypes(mtbp))
        variance_covariance!(vcov, moments, mtbp.state)
        for i in 1:2
            p = qqplot(Normal(mu[i], sqrt(vcov[i,i])), particles[t][:,i], 
                ylabel=i==1 ? "t=$t\nSample Quantiles" : "", 
                xlabel=t==100 ? "Theoretical Quantiles" : "",
                title=t==20 ? (i==1 ? "Exposed" : "Infectious") : "",
                grid=nothing)
            push!(plots, p)
        end
    end
    plot(plots...; layout = (3, 2), size=(500,600))
end
qq = makeqq(particles, mtbp)
savefig(qq, joinpath("figs", "SEIRToyModelQQPlot.pdf"))

tmp = joinpath(pwd(), "gr-temp")
println("Press any key to delete temporary directory $(tmp) and contents.")
readline(stdin)
rm(tmp; force=true, recursive=true)