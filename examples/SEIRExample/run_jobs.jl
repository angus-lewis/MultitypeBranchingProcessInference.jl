
argv = ARGS
argc = length(argv)
if argc != 2
    error("run_jobs.jl program expects 2 arguments \
            \n    - directory of jobs files to run \
            \n    - max number of jobs to run simultaneously.")
end

function run_jobs(jobs_list, script_name, max_jobs = 1)
    tasks = Dict()
    total_jobs = length(jobs_list)
    job_count = 0
    for job_file in jobs_list
        job_count += 1
        if endswith(job_file, ".yaml") 
            while length(tasks) >= max_jobs
                sleep(0.1)
                for (job, tsk) in tasks
                    if istaskdone(tsk)
                        println("Job finished: $(job)")
                        pop!(tasks, job)
                    end
                end
            end
            job = "julia --project=../. $(script_name) $(joinpath(argv[1], job_file))"
            println("Starting job: $(job_count) of $(total_jobs)")
            println("\t job name: $(job)")
            task = @async run(`julia --project=../. $(script_name) $(joinpath(argv[1], job_file))`)
            tasks[job] = task
        end
    end
    while length(tasks) > 0
        sleep(0.1)
        for (job, tsk) in tasks
            if istaskdone(tsk)
                println("Job finished: $(job)")
                pop!(tasks, job)
            end
        end
    end
end
sim_jobs = readdir(argv[1])
sim_jobs = sim_jobs[startswith.(sim_jobs, "se1i1")]
run_jobs(sim_jobs, "simulate.jl", parse(Int, argv[2]))
run_jobs(readdir(argv[1]), "inference.jl", parse(Int, argv[2]))