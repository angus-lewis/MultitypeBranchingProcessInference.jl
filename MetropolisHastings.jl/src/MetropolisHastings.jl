module MetropolisHastings

using Random
using Distributions
using LinearAlgebra
using Dates

import Random.rand
import Distributions.logpdf

include("io.jl")
include("proposal.jl")
include("buffer.jl")

export MHConfig,
    metropolis_hastings,
    SymmetricProposalDistribution,
    MutableMvNormal,
    read_binary_array_file

struct MHConfig{F<:Real, SIO<:IO, IIO<:IO, MIO<:IO}
    samples_buffer_size::Int
    samples_file::String
    samples_io::SIO
    maxiters::Int
    nparams::Int
    max_time_sec::Float64
    init_sample::Vector{F}
    verbose::Bool
    info_file::String
    info_io::IIO
    adaptive::Bool
    nadapt::Int
    adapt_cov_scale::F
    continue_from_write_file::Bool
    model_info_file::String
    model_info_io::MIO
end

function MHConfig(
    samples_buffer_size,
    samples_file::String,
    maxiters,
    nparams,
    max_time_sec,
    init_sample,
    verbose,
    info_file::String,
    adaptive,
    nadapt,
    adapt_cov_scale,
    continue_from_write_file,
    model_info_file::String="devnull"
)
    if continue_from_write_file
        if !isfile(samples_file)
            error("MH IO file does not exist: $(samples_file).")
        end
    else
        if (
            (samples_file!="devnull" && isfile(samples_file)) 
            || (info_file!="devnull" && info_file!="stdout" && isfile(info_file))
            || (model_info_file!="devnull" && isfile(model_info_file))
        )
            error("MH IO file(s) already exist\n    $(samples_file)\n    $(info_file)\n    $(model_info_file).")
        end
    end
    samples_io = if samples_file=="devnull"
        devnull
    else
        open(samples_file, "a")
    end
    info_io = if info_file=="devnull"
        devnull
    elseif info_file=="stdout"
        stdout
    else
        open(info_file, "a")
    end
    model_info_io = if model_info_file=="devnull"
        devnull
    else
        open(model_info_file, "a")
    end
    return MHConfig(
        samples_buffer_size,
        samples_file,
        samples_io,
        maxiters,
        nparams,
        max_time_sec,
        init_sample,
        verbose,
        info_file,
        info_io,
        adaptive,
        nadapt,
        adapt_cov_scale,
        continue_from_write_file,
        model_info_file,
        model_info_io,
    )
end

function printconfig(io, mh_config)
    println(io, "[INFO]     MH Config:")
    println(io, "[INFO]         samples_buffer_size: ", mh_config.samples_buffer_size)
    println(io, "[INFO]         samples_file: ", mh_config.samples_file)
    println(io, "[INFO]         maxiters: ", mh_config.maxiters)
    println(io, "[INFO]         nparams: ", mh_config.nparams)
    println(io, "[INFO]         max_time_sec: ", mh_config.max_time_sec)
    println(io, "[INFO]         init_sample: ", mh_config.init_sample)
    println(io, "[INFO]         verbose: ", mh_config.verbose)
    println(io, "[INFO]         info_file: ", mh_config.info_file)
    println(io, "[INFO]         adaptive: ", mh_config.adaptive)
    println(io, "[INFO]         nadapt: ", mh_config.nadapt)
    println(io, "[INFO]         adapt_cov_scale: ", mh_config.adapt_cov_scale)
    println(io, "[INFO]         continue_from_write_file: ", mh_config.continue_from_write_file)
    println(io, "[INFO]         model_info_file: ", mh_config.model_info_file)
    return
end

function _printinfo(verbose, info_io, samples_count, samples_buffer, start_time_sec, max_time_sec, maxsamples, loglike, inf_loglike_count)
    if !verbose
        return
    end
    accratio = samples_buffer.accepted_count/samples_buffer.bufferidx
    println(info_io, "[INFO] Iteration $(samples_count).")
    println(info_io, "[INFO] Current time $(Dates.now()).")
    println(info_io, "[INFO] Elapsed time $(elapsedtime(start_time_sec)) seconds.")
    println(info_io, "[INFO] Acceptance ratio $(accratio).")
    println(info_io, "[INFO] Number of proposed samples with infinite loglikelihood: $(inf_loglike_count) of the last $(samples_buffer.bufferidx) iterations.")
    println(info_io, "[INFO] Remaining time $(max_time_sec-(time()-start_time_sec)) seconds")
    println(info_io, "[INFO] Remaining iterations $(maxsamples-samples_count)")
    println(info_io, "[INFO] Current sample $(viewlatest(samples_buffer)).")
    println(info_io, "[INFO] Current loglikelihood $(loglike).")
    if accratio == zero(accratio)
        println(info_io, "[WARN] None of the previous $(samples_buffer.bufferidx) samples were accepted.")
    end
    flush(info_io)
    return 
end

function _printinfo_endstatus(verbose, info_io, start_time_sec, mh_config, samples_count, maxsamples, samples_io, write_samples, totalinfiniteloglikelihoodscount, unique_samples_count)
    if !verbose
        return 
    end

    if write_samples 
        header_size = length((mh_config.nparams, samples_count))+1 # add 1 for dim field in file which is not part of header
        header_size_nbytes = header_size*sizeof(Int64)
    
        end_of_samples_pos, eof_pos = position(samples_io), position(seekend(samples_io))
        samples_size_nbytes = end_of_samples_pos - header_size_nbytes
        samples_size = (samples_size_nbytes/sizeof(eltype(mh_config.init_sample))/mh_config.nparams)
        total_size_nbytes = samples_size_nbytes + header_size_nbytes
    end

    println(info_io, "[INFO] END STATUS")
    println(info_io, "[INFO]     current time $(Dates.now()).")
    println(info_io, "[INFO]     elapsed time: $(elapsedtime(start_time_sec)) seconds.")
    println(info_io, "[INFO]     timeout: $(timeout(start_time_sec, mh_config.max_time_sec)).")
    println(info_io, "[INFO]     max iters reached: $(maxitersreached(samples_count, maxsamples)).")
    println(info_io, "[INFO]     number of unique samples: $(unique_samples_count).")
    println(info_io, "[INFO]     number of proposed samples with infinite loglikelihood: $(totalinfiniteloglikelihoodscount)")
    println(info_io, "[INFO]     samples filesize:")
    write_samples && println(info_io, "[INFO]         header: $header_size ($header_size_nbytes bytes).")
    write_samples && println(info_io, "[INFO]         samples: $samples_size ($samples_size_nbytes bytes).")
    write_samples && println(info_io, "[INFO]         total: $total_size_nbytes bytes (eof at $(eof_pos)).")
    printconfig(info_io, mh_config)
    return 
end

function elapsedtime(starttime)
    return time()-starttime
end
function timeout(starttime, max_time)
    return elapsedtime(starttime) >= max_time
end

function maxitersreached(iteration, max_iterations)
    return iteration >= max_iterations
end

function adaptmaybe!(symmetric_proposal_distribution, mh_config, info_io, samples_count, samples_buffer)
    if mh_config.adaptive && samples_count<=mh_config.nadapt
        mh_config.verbose && println(info_io, "[INFO] Adapting proposal at iteration $(samples_count).")
        adapt!(symmetric_proposal_distribution, samples_buffer.buffer, mh_config.adapt_cov_scale)
        mh_config.verbose && println(info_io, "[INFO] Proposal at iteration $(samples_count): $(symmetric_proposal_distribution)")
        flush(info_io)
    end
    return 
end

function calc_log_accept_ratio(sample, loglikelihood_fn::Function, prior_logpdf::Function)
    logpdf_prior = prior_logpdf(sample)
    if isinf(logpdf_prior)
        # skip expensive likelihood evaluation when logpdf_prior = -Inf
        loglikelihood_value = logpdf_prior
        log_accept_ratio = logpdf_prior
    else
        loglikelihood_value = loglikelihood_fn(sample)
        log_accept_ratio = loglikelihood_value + logpdf_prior
    end
    return log_accept_ratio, loglikelihood_value
end

function read_init_sample_from_end_of_file!(mh_config)
    return open(mh_config.samples_file, "r") do io 
        seekend(io)
        skip(io, -sizeof(eltype(mh_config.init_sample))*mh_config.nparams)
        for i in 1:mh_config.nparams
            mh_config.init_sample[i] = read(io, eltype(mh_config.init_sample))
        end
        read_binary_array_file_header(io)
    end
end

function check_init_params(init_params, logpdf_fn::Function)
    logpdf_prior_value = logpdf_fn(init_params)
    if isinf(logpdf_prior_value) && logpdf_prior_value<zero(logpdf_prior_value)
        error("Initial params are not in the support of the prior.")
    end
    return 
end

# TODO: deprecate this implementation
function metropolis_hastings(rng::AbstractRNG, loglikelihood_fn::Function, prior_logpdf_fn::Function, symmetric_proposal_distribution, mh_config::MHConfig, close_io::Bool=true)
    # set up parameters in cases when we do or do not continue from a file
    samples_count = init_sample_setup!(mh_config)
    maxsamples = samples_count + mh_config.maxiters

    # set up first sample from config
    current_sample = mh_config.init_sample
    check_init_params(current_sample, prior_logpdf_fn)
    setstate!(symmetric_proposal_distribution, current_sample)
    infiniteloglikelihoodscount = 0
    current_log_accept_ratio, current_loglikelihood_value = 
        calc_log_accept_ratio(current_sample, loglikelihood_fn, prior_logpdf_fn)
    infiniteloglikelihoodscount += isinf(current_loglikelihood_value)

    # allocate space for samples
    proposed_sample = Vector{eltype(mh_config.init_sample)}(undef, mh_config.nparams)
    samples_buffer = SamplesBuffer(mh_config.init_sample, mh_config.samples_buffer_size, current_log_accept_ratio)

    # the file is empty and we need to leave space for the header
    if position(mh_config.samples_io)==0
        skip_binary_array_file_header(mh_config.samples_io, length(size(samples_buffer.buffer)))
    end 
    # else, we assume there is already space for the header
    
    totalinfiniteloglikelihoodscount = 0
    unique_samples_count = 0

    # continue the MCMC chain until timeout or maxiters reached
    start_time_sec = time()
    while !timeout(start_time_sec, mh_config.max_time_sec) && !maxitersreached(samples_count, maxsamples)
        # propose
        proposed_sample .= rand(rng, symmetric_proposal_distribution)
        proposed_log_accept_ratio, proposed_loglikelihood_value = 
            calc_log_accept_ratio(proposed_sample, loglikelihood_fn, prior_logpdf_fn)
        infiniteloglikelihoodscount += isinf(proposed_loglikelihood_value)
        log_accept_ratio = proposed_log_accept_ratio - current_log_accept_ratio
        
        # accept/reject
        samples_count += 1
        if log(rand(rng)) <= log_accept_ratio
            current_sample = addsample!(samples_buffer, proposed_sample, proposed_log_accept_ratio)
            setstate!(symmetric_proposal_distribution, current_sample)
            current_log_accept_ratio, current_loglikelihood_value = proposed_log_accept_ratio, proposed_loglikelihood_value
        else 
            repeatsample!(samples_buffer)
        end

        # write buffer 
        if isbufferfull(samples_buffer)
            _printinfo(mh_config.verbose, mh_config.info_io, samples_count, samples_buffer, start_time_sec, mh_config.max_time_sec, maxsamples, current_loglikelihood_value, infiniteloglikelihoodscount)

            totalinfiniteloglikelihoodscount += infiniteloglikelihoodscount
            infiniteloglikelihoodscount = 0
            unique_samples_count += samples_buffer.accepted_count

            adaptmaybe!(symmetric_proposal_distribution, mh_config, mh_config.info_io, samples_count, samples_buffer)
            writebuffer!(mh_config.samples_io, samples_buffer)
        end
    end
    _printinfo(mh_config.verbose, mh_config.info_io, samples_count, samples_buffer, start_time_sec, mh_config.max_time_sec, maxsamples, current_loglikelihood_value, infiniteloglikelihoodscount)
    # write remaing samples from buffer
    writebuffer!(mh_config.samples_io, samples_buffer)
    _printinfo_endstatus(mh_config.verbose, mh_config.info_io, start_time_sec, mh_config, samples_count, maxsamples, position(mh_config.samples_io), position(seekend(mh_config.samples_io)), totalinfiniteloglikelihoodscount, unique_samples_count)
    # now that the number of samples is known, add the file header
    header = (mh_config.nparams+1, samples_count)
    write_binary_array_file_header(mh_config.samples_io, header)

    close_io && close_ios(mh_config)

    return samples_count
end

function close_ios(mh_config::MHConfig)
    mh_config.samples_io !== stdout && close(mh_config.samples_io)
    mh_config.info_io !== stdout && close(mh_config.info_io)
    mh_config.model_info_io !== stdout && close(mh_config.model_info_io)
    return 
end

function calc_log_accept_ratio(sample, model, prior)
    logpdf_prior = logpdf(prior, sample)
    if isinf(logpdf_prior)
        # skip expensive likelihood evaluation when logpdf_prior = -Inf
        loglikelihood_value = logpdf_prior
        log_accept_ratio = logpdf_prior
    else
        loglikelihood_value = logpdf(model, sample)
        log_accept_ratio = loglikelihood_value + logpdf_prior
    end
    return log_accept_ratio, loglikelihood_value
end

function init_sample_setup!(mh_config)
    if mh_config.continue_from_write_file
        header = read_init_sample_from_end_of_file!(mh_config) 
        samples_count = header[2]
    else
        samples_count = 0
    end
    return samples_count
end

function check_init_params(init_params, prior)
    logpdf_prior_value = logpdf(prior, init_params)
    if isinf(logpdf_prior_value) && logpdf_prior_value<zero(logpdf_prior_value)
        error("Initial params are not in the support of the prior.")
    end
    return 
end

function write_model_info(io::IO, model, isacc::Bool)
    error("Implementation specific model info writer not found, perhaps it has not been implemented.")
end

function metropolis_hastings(rng::AbstractRNG, model, prior, proposal, mh_config::MHConfig, write_samples::Bool=true, close_io::Bool=true)
    # set up parameters in cases when we do or do not continue from a file
    samples_count = init_sample_setup!(mh_config)
    maxsamples = samples_count + mh_config.maxiters

    # determine if we are to write model info (requires write_model_info(io, model, isaccepted) function to be defined)
    do_write_model_info = !(mh_config.model_info_io === devnull)

    # set up first sample from config
    current_sample = mh_config.init_sample
    check_init_params(current_sample, prior)
    setstate!(proposal, current_sample)
    inf_loglike_count = 0
    current_log_accept_ratio, current_loglike_value = calc_log_accept_ratio(current_sample, model, prior)
    inf_loglike_count += isinf(current_loglike_value)
    # write model info for initial sample
    do_write_model_info && write_model_info(mh_config.model_info_io, model, true)

    # allocate space for samples
    proposed_sample = Vector{eltype(mh_config.init_sample)}(undef, mh_config.nparams)
    samples_buffer = SamplesBuffer(mh_config.init_sample, mh_config.samples_buffer_size, current_log_accept_ratio)
    samples_count += 1

    # the file is empty and we need to leave space for the header
    if write_samples
        samples_io = mh_config.samples_io
        if position(mh_config.samples_io)==0
            skip_binary_array_file_header(samples_io, length(size(samples_buffer.buffer)))
        end 
        # else, we assume there is already space for the header
    else
        samples_io = SamplesBuffer(mh_config.init_sample, mh_config.maxiters, current_log_accept_ratio, true)
    end
    
    total_inf_loglike_count = 0
    unique_samples_count = 0

    # continue the MCMC chain until timeout or maxiters reached
    start_time_sec = time()
    while !timeout(start_time_sec, mh_config.max_time_sec) && !maxitersreached(samples_count, maxsamples)
        # propose
        proposed_sample .= rand(rng, proposal)
        proposed_log_accept_ratio, proposed_loglike_value = calc_log_accept_ratio(proposed_sample, model, prior)
        inf_loglike_count += isinf(proposed_loglike_value)
        log_accept_ratio = proposed_log_accept_ratio - current_log_accept_ratio
        
        # accept/reject
        samples_count += 1
        if log(rand(rng)) <= log_accept_ratio
            current_sample = addsample!(samples_buffer, proposed_sample, proposed_log_accept_ratio)
            setstate!(proposal, current_sample)
            current_log_accept_ratio, current_loglike_value = proposed_log_accept_ratio, proposed_loglike_value
            write_samples && do_write_model_info && write_model_info(mh_config.model_info_io, model, true)
        else 
            repeatsample!(samples_buffer)
            write_samples && do_write_model_info && write_model_info(mh_config.model_info_io, model, false)
        end

        # write buffer 
        if isbufferfull(samples_buffer)
            _printinfo(mh_config.verbose, mh_config.info_io, samples_count, samples_buffer, start_time_sec, mh_config.max_time_sec, maxsamples, current_loglike_value, inf_loglike_count)

            total_inf_loglike_count += inf_loglike_count
            inf_loglike_count = 0
            unique_samples_count += samples_buffer.accepted_count

            adaptmaybe!(proposal, mh_config, mh_config.info_io, samples_count, samples_buffer)
            writebuffer!(samples_io, samples_buffer)
        end
    end
    _printinfo(mh_config.verbose, mh_config.info_io, samples_count, samples_buffer, start_time_sec, mh_config.max_time_sec, maxsamples, current_loglike_value, inf_loglike_count)
    # write remaing samples from buffer
    writebuffer!(samples_io, samples_buffer)
    _printinfo_endstatus(mh_config.verbose, mh_config.info_io, start_time_sec, mh_config, samples_count, maxsamples, samples_io, write_samples, total_inf_loglike_count, unique_samples_count)
    # now that the number of samples is known, add the file header, nrows=nparams+1 to include the loglike
    header = (mh_config.nparams+1, samples_count)
    write_samples && write_binary_array_file_header(samples_io, header)

    write_samples && close_io && close_ios(mh_config)

    if write_samples
        return samples_count
    end
    return samples_io.buffer
end

end
