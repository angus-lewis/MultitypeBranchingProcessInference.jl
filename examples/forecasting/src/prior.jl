function makepriordist(mu, covfuntype::String, covfunparams, R0changepoints)
    if covfuntype=="exponential"
        cov_fun = GP.ExponentialCovarianceFunction(covfunparams...)
        timestamps = Matrix(reshape(Float64.(R0changepoints), 1, :))
        R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
    elseif covfuntype=="squared_exponential"
        cov_fun = GP.SquaredExponentialCovarianceFunction(covfunparams...)
        timestamps = Matrix(reshape(Float64.(R0changepoints), 1, :))
        R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
    else 
        error("Unknown covariance function in config")
    end
    return R0prior
end

function makeprior(mu, covfuntype::String, covfunparams, R0changepoints, transform)
    R0prior = makepriordist(mu, covfuntype, covfunparams, R0changepoints)

    transform_fn = transform=="log" ? log : identity
    log_jacobian_fn = transform=="log" ? ((x) -> -sum(x)) : (x -> zero(x))

    transformed_params = zeros(Float64, size(R0prior.x, 2))
    gpmemcache = GP.gp_logpdf_memcache(R0prior, transformed_params)

    prior_logpdf = (params) -> begin
        if (transform=="log") && any(p -> p <= zero(p), params)
            return -Inf
        end
        for i in eachindex(params)
            transformed_params[i] = transform_fn(params[i])
        end
        val = GP.logpdf(R0prior, transformed_params, gpmemcache)
        val += log_jacobian_fn(transformed_params)
        return val
    end
    return prior_logpdf
end