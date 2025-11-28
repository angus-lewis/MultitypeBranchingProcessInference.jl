function extract_run_time_sec(info_file_io)
    key = "[INFO]     elapsed time: "
    for line in eachline(info_file_io)
        if length(line)>length(key) && line[1:length(key)] == key
            runtime = split(line[length(key)+1:end], " ")[1]
            return runtime
        end
    end
    return 
end

function extract_burnin_time_sec(info_file_io)
    last_burnin_key = "[INFO] Iteration 20480."
    is_last_burn_in_iter = false
    time_key = "[INFO] Elapsed time "
    for line in eachline(info_file_io)
        if length(line)>=length(last_burnin_key) && line[1:length(last_burnin_key)] == last_burnin_key
            is_last_burn_in_iter = true
        end
        if (is_last_burn_in_iter 
            && length(line)>length(time_key) 
            && line[1:length(time_key)] == time_key)
            runtime = split(line[length(time_key)+1:end], " ")[1]
            return runtime
        end
    end
    return 
end

function extract_summary_stats(summary_file_io)
    ss_key = "Summary Statistics"
    for line in eachline(summary_file_io)
        if length(line) >= length(ss_key) && line[1:length(ss_key)] == ss_key
            stats_names = readline(summary_file_io)
            stats_names = split(stats_names, Regex("\\s+"))
            stats_dtypes = readline(summary_file_io) # skip this line 
            _ = readline(summary_file_io) # skip this line 
            stats_l = readline(summary_file_io)
            stats_v = split(stats_l, Regex("\\s+"))
            stats_d = Dict(stats_names .=> stats_v)
            return stats_d
        end
    end
    return 
end

function extract_data(dir, n, R0, T, S, p, method)
    infofile = joinpath(dir, "infose$(n)i$(n)r_infection_rate=$(R0)_nsteps=$(T)_cov=[$(S)]_observation_probability=$(p)_method=$(method).txt")
    runtime = open(infofile, "r") do io
        return extract_run_time_sec(io)
    end
    burnin_time = open(infofile, "r") do io
        return extract_burnin_time_sec(io)
    end

    summaryfile = joinpath(dir, "samplesse$(n)i$(n)r_infection_rate=$(R0)_nsteps=$(T)_cov=[$(S)]_observation_probability=$(p)_method=$(method).f64_array.bin.summary.txt")
    summary = open(summaryfile, "r") do io
        return extract_summary_stats(io)
    end
    summary["full time"] = runtime
    summary["burnin time"] = burnin_time
    summary["time"] = string(parse(Float64, runtime) - parse(Float64, burnin_time))
    return summary
end

dir = "."
n = [1; 8]
r = [0.12; 0.3; 0.5]
s = [0.25; 1.0; 4.0]
T = [10; 15; 25]
p = [0.5; 0.75; 1.0]
m = ["kalman_filter"; "hybrid"; "particle_filter"]

mmap = Dict( m .=> ["Gaussian"; "Hybrid"; "Particle"])
rnd(x) = x isa AbstractFloat ? round(x,digits=1) : x
tbl_row(di, s, m) = "$(s) & $(mmap[m]) & $(rnd(parse(Float64,di["ess_bulk"]))) & $(rnd(parse(Float64, di["ess_bulk"])/parse(Float64, di["time"]))) & $(rnd(parse(Float64, di["time"]))) \\\\ "

c = 0 
for ii in [1;8]
    c += 1
    # SE1I1R R0 experiment 
    table_str = "\$R_0\$ & Method & ESS & ESS/sec & Time (s)\\\\ \\hline \\hline \n"
    rn = 0
    for ni in n[c]
        for ri in r
            for Ti in T[3]
                for Si in s[2]
                    for pi in p[2]
                        for mi in m
                            di = extract_data(dir, ni, ri, Ti, Si, pi, mi)
                            table_str *= tbl_row(di, round(ri/(3/28), digits=2), mi)
                            rn += 1
                            if rn%3==0
                                table_str *= "\\hline \n"
                            else
                                table_str *= "\n"
                            end
                        end
                    end
                end
            end
        end
    end
    open("SE$(ii)I$(ii)R_R0_experiment_stats.txt", "w") do io
        print(io, table_str)
    end

    # SE1I1R T experiment 
    table_str = "\$T\$ & Method & ESS & ESS/sec & Time (s)\\\\ \\hline \\hline \n"
    rn = 0
    for ni in n[c]
        for ri in r[2]
            for Ti in T
                for Si in s[2]
                    for pi in p[2]
                        for mi in m
                            @show di = extract_data(dir, ni, ri, Ti, Si, pi, mi)
                            table_str *= tbl_row(di, Ti, mi)
                            rn += 1
                            if rn%3==0
                                table_str *= "\\hline \n"
                            else
                                table_str *= "\n"
                            end
                        end
                    end
                end
            end
        end
    end
    open("SE$(ii)I$(ii)R_T_experiment_stats.txt", "w") do io
        print(io, table_str)
    end

    # SE1I1R p experiment 
    table_str = "\$p\$ & Method & ESS & ESS/sec & Time (s)\\\\ \\hline \\hline \n"
    rn = 0
    for ni in n[c]
        for ri in r[2]
            for Ti in T[3]
                for Si in s[2]
                    for pi in p
                        for mi in m
                            @show di = extract_data(dir, ni, ri, Ti, Si, pi, mi)
                            table_str *= tbl_row(di, pi, mi)
                            rn += 1
                            if rn%3==0
                                table_str *= "\\hline \n"
                            else
                                table_str *= "\n"
                            end
                        end
                    end
                end
            end
        end
    end
    open("SE$(ii)I$(ii)R_p_experiment_stats.txt", "w") do io
        print(io, table_str)
    end

    # SE1I1R S experiment 
    table_str = "\$\\sigma^2\$ & Method & ESS & ESS/sec & Time (s)\\\\ \\hline \\hline \n"
    rn = 0
    for ni in n[c]
        for ri in r[2]
            for Ti in T[3]
                for Si in s
                    for pi in p[2]
                        for mi in m
                            @show di = extract_data(dir, ni, ri, Ti, Si, pi, mi)
                            table_str *= tbl_row(di, Si, mi)
                            rn += 1
                            if rn%3==0
                                table_str *= "\\hline \n"
                            else
                                table_str *= "\n"
                            end
                        end
                    end
                end
            end
        end
    end
    open("SE$(ii)I$(ii)R_S_experiment_stats.txt", "w") do io
        print(io, table_str)
    end
end
