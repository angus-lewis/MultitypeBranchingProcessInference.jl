using YAML

argv = ARGS
argc = length(argv)
if argc < 2
    error("generate_configs.jl program expects 2 arguments \
            \n    - master config file name
            \n.   - list of experiments files with the parameters to change.")
end

config = YAML.load_file(joinpath(pwd(), ARGS[1]))

experiments = []
for i in 2:argc
    push!(experiments, YAML.load_file(joinpath(pwd(), ARGS[i])))
end

function generate_configs!(master_config, experiments, 
    name = "se$(master_config["model"]["stateprocess"]["params"]["E_state_count"])i$(master_config["model"]["stateprocess"]["params"]["I_state_count"])r")
    if isempty(experiments)
        simname = "simse1i1"*name[6:end]
        master_config["simulation"]["outfilename"] = joinpath("data", "$(simname).Int64.bin")
        master_config["inference"]["mh_config"]["outfilename"] = joinpath("data", "samples$(name).f64_array.bin")
        master_config["inference"]["mh_config"]["infofilename"] = joinpath("data", "info$(name).txt")
        if occursin("kalman", name) || occursin("particle", name)
            master_config["inference"]["mh_config"]["model_info_filename"] = joinpath("data", "model_info$(name).f64_array.bin")
        else
            master_config["inference"]["mh_config"]["model_info_filename"] = "nothing"
        end
        YAML.write_file(joinpath(pwd(), "experiments", "config$(name).yaml"), master_config)
        return 
    end

    experiment = pop!(experiments).second

    key_list = experiment["keys"]
    for value in experiment["values"]
        # iterate to leaf of yaml and update it
        elt = master_config
        for key in key_list[1:end-1]
            elt = elt[key]
        end
        elt[key_list[end]] = value
        newname = name * "_" * key_list[end] * "=" * string(value)
        generate_configs!(master_config, deepcopy(experiments), newname)
    end

    return 
end

for experiment in experiments
    generate_configs!(config, experiment)
end


