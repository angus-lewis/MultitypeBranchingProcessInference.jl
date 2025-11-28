using StatsPlots
using YAML
using Distributions
using LaTeXStrings
using MultitypeBranchingProcessInference
using LinearAlgebra

const PLOTTIMES = [5; 15; 25]

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
    
    scattertimes = PLOTTIMES
    pastel_blue = RGB(0.7, 0.8, 1.0)
    pastel_green = RGB(0.7, 1.0, 0.7)
    pastel_red = RGB(1.0, 0.7, 0.7)
    colours = [pastel_blue, pastel_green, pastel_red]
    markers = [:diamond, :utriangle, :circle]
    i = 1
    for t in scattertimes
        scatter!(particles[t][:,1], particles[t][:,2], color=colours[i], label="t=$t", markersize=3, marker=markers[i])
        i += 1
    end
    i = 1
    for t in scattertimes
        moments!(moments, mtbp, t)
        mu = mean(moments, mtbp.state)
        sigma = variance_covariance(moments, mtbp.state)
        # covellipse!(mu[1:2], sigma[1:2,1:2]; n_std = 1, color=:lightblue, label=nothing, alpha=0.4)
        covellipse!(mu[1:2], sigma[1:2,1:2]; n_std = 1, color=colours[i], label=nothing, alpha=0.4)
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
    
    return plot!(grid=nothing)#, xscale=:log10, yscale=:log10, xlims=(0.9, 200), ylims=(0.9,330))
end
samplepathplot = makeplot(particles, mtbp)
savefig(samplepathplot, joinpath("figs", "SEIRToyModelSamplePathPlot.pdf"))

function makeqq(particles, mtbp)
    moments = MTBPMomentsOperator(mtbp)
    init!(mtbp)
    plots = Plots.Plot[]
    for t in PLOTTIMES
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

function makets(particles, mtbp)
    moments = MTBPMomentsOperator(mtbp)
    init!(mtbp)

    sorted_keys = sort(collect(keys(particles)))
    series = zeros(length(particles), size(particles[1], 1)+1)
    count = 0
    for t in sorted_keys
        count += 1
        series[count, 1] = t
        series[count, 2:end] = particles[t][:,end]
    end
    pastel_red = RGB(1.0, 0.7, 0.7)
    p1 = plot(series[2:end,1], diff(series[:,2:end], dims=1); xlabel=L"\mbox{Time}", ylabel=L"\mbox{New cases: }"*" "*L"z_3(t)-z_3(t-1)", label=false, color=:grey, alpha=0.3, legend=:topleft, grid=nothing)
    p1 = scatter!(p1, series[2:end,1], diff(series[:,2], dims=1); label=L"\mbox{Sample path}", color=:grey)

    mu_series = zeros(length(particles), mtbp.ntypes+1)
    sigma_series = zeros(length(particles), mtbp.ntypes+1)
    count = 0
    t0 = first(sorted_keys)
    for t in sorted_keys
        count += 1 

        moments!(moments, mtbp, t)
        mu = zeros(paramtype(mtbp), getntypes(mtbp))
        mean!(mu, moments, mtbp.state)
        mu_series[count,1] = t
        mu_series[count, 2:end] = mu

        moments!(moments, mtbp, t-t0)
        t0 = t
        cov = zeros(paramtype(mtbp), getntypes(mtbp), getntypes(mtbp))
        mu[end] = 0
        variance_covariance!(cov, moments, mu)
        sigma_series[count,1] = t
        sigma_series[count, 2:end] = sqrt.(diag(cov))
    end
    plot!(p1, mu_series[2:end, 1], diff(mu_series[:,end]), 
        ribbon = sigma_series[2:end, end],
        label=L"E[z_3(t)-z_3(t-1)]", color=:blue, linewidth=2, fillalpha=0.2)
    # plot!(p1, sigma_series[2:end, 1], diff(sigma_series[:,end]), label=false, color=:lightblue, linewidth=2)
    # plot!(p2, mu_series[:,1], mu_series[:,3]; label=L"E[z_2(t)]", color=:black, linewidth=2)

    return plot(p1; layout=(2,1), size=(600,400))
end

ts = makets(particles, mtbp)
savefig(ts, joinpath("figs", "SEIRToyModelTSPlot.pdf"))
