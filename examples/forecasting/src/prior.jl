struct R0Prior{P<:GP.GaussianProcess, F<:AbstractFloat, T}
    gp::P
    transform::Symbol
    _malloc::Vector{F}
    _gp_malloc::T
end

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
    R0dist = makepriordist(mu, covfuntype, covfunparams, R0changepoints)
    malloc = similar(R0dist.mu)
    r0prior = R0Prior(
        R0dist,
        Symbol(transform),
        malloc,
        GP.gp_logpdf_memcache(R0dist, malloc)
    )
    return r0prior
end

function Distributions.logpdf(r0prior::R0Prior, params)
    return logpdf(r0prior, params, Val(r0prior.transform))
end

function Distributions.logpdf(r0prior::R0Prior, params, transform::Val{:log})
    if any(p -> p <= zero(p), params)
        return -Inf
    end
    for i in 1:length(params)
        r0prior._malloc[i] = log(params[i])
    end
    val = GP.logpdf(r0prior.gp, r0prior._malloc, r0prior._gp_malloc)
    val += -sum(r0prior._malloc)
    return val
end

function Distributions.logpdf(r0prior::R0Prior, params, transform::Val{:none})
    val = GP.logpdf(r0prior.gp, params, r0prior._gp_malloc)
    return val
end