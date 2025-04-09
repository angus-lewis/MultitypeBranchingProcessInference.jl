abstract type MeanModel end

mutable struct Logistic4{F<:AbstractFloat} <: MeanModel
    L::F # real L0 <= L
    L0::F # real L0 <= L
    k::F # negative (decreasing curves only)
    x0::F # real
end

function paramcount(l::Logistic4)
    return 4
end

function (f::Logistic4{F})(x) where F
    return f.L0 + (f.L-f.L0)/(F(1) + exp(-f.k*(x-f.x0)))
end

function (f::Logistic4{F})(x::AbstractArray) where F
    return f(only(x))
end

struct R0Prior{P<:GP.GaussianProcess, F<:AbstractFloat, T, S<:MeanModel, R}
    gp::P
    meanmodel::S
    meanmodelprior::R
    transform::Symbol
    _malloc::Vector{F}
    _gp_malloc::T
end

function makepriordist(meanmodelparams, covfuntype::String, covfunparams, R0changepoints)
    timestamps = Matrix(reshape(Float64.(R0changepoints), 1, :))

    mu = zeros(Float64, length(timestamps))
    meanmodel = Logistic4(meanmodelparams...)
    for i in eachindex(mu)
        mu[i] = meanmodel(timestamps[:,i])
    end

    if covfuntype=="exponential"
        cov_fun = GP.ExponentialCovarianceFunction(covfunparams...)
    elseif covfuntype=="squared_exponential"
        cov_fun = GP.SquaredExponentialCovarianceFunction(covfunparams...)
    else 
        error("Unknown covariance function in config")
    end

    R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)

    return R0prior, meanmodel
end

struct Logistic4Hyperprior{D1,D2,D3,D4}
    Ldist::D1
    L0dist::D2
    kdist::D3
    x0dist::D4
end

function makehyperprior(params)
    Ldist = Normal(params[1:2]...)
    L0dist = Gamma(params[3:4]...)
    kdist = Gamma(params[5:6]...)
    x0dist = Normal(params[7:8]...)
    return Logistic4Hyperprior(Ldist, L0dist, kdist, x0dist)
end

function Distributions.logpdf(u::Logistic4Hyperprior, x)
    if x[1] >= x[2] && x[3]<zero(eltype(x))
        val = logpdf(u.Ldist, x[1])
        val += logpdf(u.L0dist, x[1]-x[2])
        val += logpdf(u.kdist, -x[3])
        val += logpdf(u.x0dist, x[4])
        return val
    end
    return -Inf
end

function makeprior(meanmodelparams, meanmodelpriorparams, covfuntype::String, covfunparams, R0changepoints, transform)
    meanmodelprior = makehyperprior(meanmodelpriorparams)
    R0dist, meanmodel = makepriordist(meanmodelparams, covfuntype, covfunparams, R0changepoints)
    malloc = similar(R0dist.mu)
    r0prior = R0Prior(
        R0dist,
        meanmodel,
        meanmodelprior,
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
    offset = paramcount(r0prior.meanmodel)+1

    # check if there are any negative R0s
    if any(p -> p <= zero(p), @view(params[offset:end]))
        return -Inf
    end

    # evaluate mean model hyper prior logpdf
    val = logpdf(r0prior.meanmodelprior, params[1:(offset-1)])
    if isinf(val) && val < zero(val)
        return val
    end
    for i in 1:(offset-1)
        setfield!(r0prior.meanmodel, i, params[i])
    end

    # set r0 prior mean using mean model
    for i in eachindex(r0prior.gp.mu)
        r0prior.gp.mu[i] = r0prior.meanmodel(r0prior.gp.x[:,i])
    end

    # evaluate r0 prior logpdf
    for i in offset:length(params)
        r0prior._malloc[i-offset+1] = log(params[i])
    end
    val += GP.logpdf(r0prior.gp, r0prior._malloc, r0prior._gp_malloc)
    val += -sum(r0prior._malloc)
    return val
end

function Distributions.logpdf(r0prior::R0Prior, params, transform::Val{:none})
    offset = paramcount(r0prior.meanmodel)+1

    # evaluate mean model hyper prior logpdf
    val = logpdf(r0prior.meanmodelprior, params[1:(offset-1)])
    if isinf(val) && val < zero(val)
        return val
    end
    for i in 1:(offset-1)
        setfield!(r0prior.meanmodel, i, params[i])
    end

    # set r0 prior mean using mean model
    for i in eachindex(r0prior.gp.mu)
        r0prior.gp.mu[i] = r0prior.meanmodel(r0prior.gp.x[:,i])
    end

    for i in offset:length(params)
        r0prior._malloc[i-offset+1] = params[i]
    end

    val = GP.logpdf(r0prior.gp, r0prior._malloc, r0prior._gp_malloc)
    return val
end