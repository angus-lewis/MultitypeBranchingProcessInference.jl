function setenvironment!(config)
    if "env" in keys(config) && "blas_num_threads" in keys(config["env"])
        LinearAlgebra.BLAS.set_num_threads(config["env"]["blas_num_threads"])
    end
    return 
end

function read_observations(filename)
    observations = open(filename, "r") do io
        nlines = countlines(io)
        seekstart(io)

        lineno = 1
        # skip header line
        readline(io)
        
        observations = Vector{Float64}[]
        while !eof(io)
            lineno += 1
            observation_string = readline(io)
            obs = [parse(Float64, observation_string)]
            push!(observations, obs)
        end
        if lineno != nlines
            error("Bad observations file")
        end
        observations
    end
    return observations
end

function makerng(seed)
    rng = Xoshiro()
    Random.seed!(rng, seed)
    return rng
end