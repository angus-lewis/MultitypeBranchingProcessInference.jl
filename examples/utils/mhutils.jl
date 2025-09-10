function makeproposal(config)
    propconfig = config["inference"]["proposal_parameters"]
    mu = propconfig["mean"]
    sigma = reshape(propconfig["cov"], length(mu), length(mu))
    return MutableMvNormal(mu, sigma)
end

function makemhconfig(config)
    mh_rng = makerng(config["inference"]["mh_config"]["seed"])
    model_info_file = if "model_info_filename" in keys(config["inference"]["mh_config"])
        open(config["inference"]["mh_config"]["model_info_filename"], "w")
    else
        devnull
    end
    mh_config = MHConfig(
        config["inference"]["mh_config"]["buffer_size"],
        config["inference"]["mh_config"]["thin"],
        config["inference"]["mh_config"]["max_iters"],
        config["inference"]["mh_config"]["nparams"],
        config["inference"]["mh_config"]["max_time_sec"],
        config["inference"]["mh_config"]["init_sample"],
        config["inference"]["mh_config"]["verbose"],
        config["inference"]["mh_config"]["adaptive"],
        config["inference"]["mh_config"]["nadapt"],
        config["inference"]["mh_config"]["adapt_cov_scale"],
        config["inference"]["mh_config"]["continue"],
        config["inference"]["mh_config"]["model_info_thin"],
    )
    return (
        mh_rng, 
        mh_config, 
        open(config["inference"]["mh_config"]["outfilename"], "w"), 
        open(config["inference"]["mh_config"]["infofilename"], "w"), 
        model_info_file
    )
end