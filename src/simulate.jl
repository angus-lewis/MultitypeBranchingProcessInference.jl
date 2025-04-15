function simulate(rng::AbstractRNG, bp::MultitypeBranchingProcess, param_seq::MTBPParamsSequence, t::AbstractVector{<:Real}, max_pop_size::Union{Real,Nothing}=nothing)
    return simulate!(rng, zeros(variabletype(bp), getntypes(bp), length(t)), bp, param_seq, t, max_pop_size)
end

function MultitypeBranchingProcesses.simulate!(
    rng::AbstractRNG, path::AbstractArray, bp::MultitypeBranchingProcess, param_seq::MTBPParamsSequence, t::AbstractVector{<:Real}, 
    max_pop_size::Union{Real,Nothing}=nothing
)
    @assert first(param_seq).time <= first(t)  "Sample times are before the first param time"
    @assert iszero(first(t)) "First sample time must be zero, got $(first(t))"
    @assert size(path, 2)==length(t)
    @assert size(path, 1)==getntypes(bp)

    # find the index of the first set of parameters before first(t)
    param_idx = 1
    next_param_time = Inf
    while param_idx < length(param_seq)
        next_param_time = param_seq[param_idx+1].time
        if next_param_time > first(t)
            break
        end
        param_idx += 1
    end
    setparams!(bp, param_seq[param_idx])

    init!(rng, bp)
    path[:,1] .= bp.state

    prevt = first(t)
    i = 1
    for ti in Iterators.drop(t, 1)
        i += 1
        if ti==next_param_time
            param_idx += 1
            setparams!(bp, param_seq[param_idx])
            next_param_time = param_idx < length(param_seq) ? param_seq[param_idx+1].time : Inf
        elseif ti > next_param_time
            error("Parameter times must coincide with sample times")
        end
        dt = ti - prevt
        simulate!(rng, bp, dt)
        path[:,i] .= bp.state
        if max_pop_size!==nothing && any(x -> x > max_pop_size, bp.state)
            path[:,i+1:end] .= -1
            break
        end
    end
    return path
end

function meanpath!(path::AbstractArray,
     bp::MultitypeBranchingProcess, 
     param_seq::MTBPParamsSequence, 
     t::AbstractVector{<:Real}
)
    @assert first(param_seq).time <= first(t)  "Sample times are before the first param time"
    @assert iszero(first(t)) "First sample time must be zero, got $(first(t))"
    @assert size(path, 2)==length(t)
    @assert size(path, 1)==getntypes(bp)

    # find the index of the first set of parameters before first(t)
    param_idx = 1
    next_param_time = Inf
    while param_idx < length(param_seq)
        next_param_time = param_seq[param_idx+1].time
        if next_param_time > first(t)
            break
        end
        param_idx += 1
    end
    setparams!(bp, param_seq[param_idx])
    
    path[:,1] .= bp.initial_state.first_moments

    op = MTBPMomentsOperator(bp)

    prevt = first(t)
    i = 1
    for ti in Iterators.drop(t, 1)
        i += 1
        if ti==next_param_time
            param_idx += 1
            setparams!(bp, param_seq[param_idx])
            next_param_time = param_idx < length(param_seq) ? param_seq[param_idx+1].time : Inf
        elseif ti > next_param_time
            error("Parameter times must coincide with sample times")
        end
        dt = ti - prevt
        moments!(op, bp, dt)
        @views mean!(path[:,i], op, path[:,i-1])
    end
    return path
end