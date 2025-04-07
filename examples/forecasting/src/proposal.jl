function makeforecastingproposal(config)
    propconfig = config["inference"]["proposal_parameters"]
    mu = propconfig["mean"]
    sigma = reshape(propconfig["cov"], length(mu), length(mu))
    return MetropolisHastings.MutableMvNormal(mu, sigma)
end