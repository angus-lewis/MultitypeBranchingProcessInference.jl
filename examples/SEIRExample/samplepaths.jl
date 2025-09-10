using YAML
using StatsPlots

include("./utils/io.jl")
include("./utils/figs.jl")

# SEIR
lowpath, t = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.024_nsteps=90_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
medpath, t100 = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.04_nsteps=90_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
highpath, t = readparticles(joinpath(pwd(), "data/simse1i1r_infection_rate=0.06_nsteps=90_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
ts = [t, t100, t]

exposed = Vector{Vector{Int64}}(undef, 3)
exposed[1] = [lowpath[i][1] for i in 1:length(t)]
exposed[2] = [medpath[i][1] for i in 1:length(t100)]
exposed[3] = [highpath[i][1] for i in 1:length(t)]

infectious = Vector{Vector{Int64}}(undef, 3)
infectious[1] = [lowpath[i][2] for i in 1:length(t)]
infectious[2] = [medpath[i][2] for i in 1:length(t100)]
infectious[3] = [highpath[i][2] for i in 1:length(t)]

totals30 = [sum(exposed[i][1:31]+infectious[i][1:31]) for i in 1:3]
totals60 = [sum(exposed[i][1:61]+infectious[i][1:61]) for i in 1:3]
totals90 = [sum(exposed[i][1:91]+infectious[i][1:91]) for i in 1:3]

new_infectious = Vector{Vector{Int64}}(undef, 3)
new_infectious[1] = [lowpath[i][3] for i in 1:length(t)]
new_infectious[2] = [medpath[i][3] for i in 1:length(t100)]
new_infectious[3] = [highpath[i][3] for i in 1:length(t)]

p1 = plot(size=(600,400)); p2 = plot(size=(600,400)); p3 = plot(size=(600,400))
colours = [_pastel_blue, _pastel_green, _pastel_red]
labels = ["R0 = $(0.024/0.02142857142857143)" "R0 = $(round(0.04/0.02142857142857143, digits=2))" "R0 = $(0.06/0.02142857142857143)"]
for i in 1:3
    p1 = plot!(p1, ts[i], exposed[i]; ylabel="Exposed", legend=:topleft,
        label=labels[i],
        color=colours[i], linewidth=2)
    p2 = plot!(p2, ts[i], infectious[i]; ylabel="Infectious", legend=false, color=colours[i], linewidth=2)
    p3 = plot!(p3, ts[i][2:end], diff(new_infectious[i], dims=1); xlabel="Time (days)", ylabel="Observed", legend=false, color=colours[i], linewidth=2)
end
yl1 = ylims(p1)
yl2 = ylims(p2)
yl = (0, max(yl1[2], yl2[2]))
p1 = plot!(p1, ylim=yl)
p2 = plot!(p2, ylim=yl)
p = plot!(p1, p2, p3, layout=(3,1), size=(600,800))

savefig(p, "figs/seir_samplepaths.pdf")


# SEIR
lowpath, t = readparticles(joinpath(pwd(), "data/simse4i4r_infection_rate=0.024_nsteps=90_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
medpath, t = readparticles(joinpath(pwd(), "data/simse4i4r_infection_rate=0.04_nsteps=90_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))
highpath, t = readparticles(joinpath(pwd(), "data/simse4i4r_infection_rate=0.06_nsteps=90_cov=[1.0]_observation_probability=0.75_method=particle_filter.Int64.bin"))

exposed = zeros(length(t), 3)
exposed[:, 1] = [sum(lowpath[i][1:4]) for i in 1:length(t)]
exposed[:, 2] = [sum(medpath[i][1:4]) for i in 1:length(t)]
exposed[:, 3] = [sum(highpath[i][1:4]) for i in 1:length(t)]

infectious = zeros(length(t), 3)
infectious[:, 1] = [sum(lowpath[i][5:8]) for i in 1:length(t)]
infectious[:, 2] = [sum(medpath[i][5:8]) for i in 1:length(t)]
infectious[:, 3] = [sum(highpath[i][5:8]) for i in 1:length(t)]

new_infectious = zeros(length(t), 3)
new_infectious[:, 1] = [lowpath[i][9] for i in 1:length(t)]
new_infectious[:, 2] = [medpath[i][9] for i in 1:length(t)]
new_infectious[:, 3] = [highpath[i][9] for i in 1:length(t)]

p1 = plot(t, exposed; ylabel="Exposed", legend=:topleft, 
    label=["R0 = $(0.024/0.02142857142857143)" "R0 = $(round(0.04/0.02142857142857143, digits=2))" "R0 = $(0.06/0.02142857142857143)"], 
    color=[_pastel_blue _pastel_green _pastel_red], linewidth=2,
    size=(600,400))
p2 = plot(t, infectious; ylabel="Infectious", legend=false, color=[_pastel_blue _pastel_green _pastel_red], linewidth=2, size=(600,400))
p3 = plot(t[2:end], diff(new_infectious, dims=1); xlabel="Time (days)", ylabel="Observed", legend=false, color=[_pastel_blue _pastel_green _pastel_red], linewidth=2, size=(600,400))
yl1 = ylims(p1)
yl2 = ylims(p2)
yl = (0, max(yl1[2], yl2[2]))
p1 = plot!(p1, ylim=yl)
p2 = plot!(p2, ylim=yl)
p = plot!(p1, p2, p3, layout=(3,1), size=(600,800))

savefig(p, "figs/seninr_samplepaths.pdf")