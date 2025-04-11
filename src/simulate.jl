function simulate(bp::MultitypeBranchingProcess, param_seq::MTBPParamsSequence, t::AbstractVector{<:Real})
    return simulate!(zeros(variabletype(bp), getntypes(bp), length(t)), bp, param_seq, t)
end

function simulate!(path::AbstractArray, bp::MultitypeBranchingProcess, param_seq::MTBPParamsSequence, t::AbstractVector{<:Real})
    param_time = first(param_seq).time
    @assert param_time <= first(t)  "Sample times are before the first param time"
    @assert first(t)==zero(t) "First sample time must be zero"

    param_idx = nothing
    next_param_time = nothing
    for param_idx in eachindex(param_seq)
        param_time = param_seq[param_idx].time
        if param_time >= first(t)
            param_idx -= 1
            next_param_time = param_seq[param_idx+1].time
            break
        end
    end
    if next_param_time===nothing 
        next_param_time = Inf
    end

    init!(rng, bp)
    path[:,1] .= bp.state

    setparams!(bp, param_seq[param_idx])
    prevt = first(t)
    i = 1
    for ti in Iterators.drop(t, 1)
        i += 1
        if ti==next_param_time
            param_idx += 1
            setparams!(bp, param_seq[param_idx])
            next_param_time = param_seq[param_idx].time
        elseif ti > next_param_time
            error("Parameter times must coincide with sample times")
        end
        dt = ti - prevt
        simulate!(rng, bp, dt)
        path[:,i] .= bp.state
    end
    return path
end