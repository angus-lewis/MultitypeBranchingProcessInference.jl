mutable struct SamplesBuffer{T}
    const buffer::T
    bufferidx::Int
    accepted_count::Int
    const buffersize::Int
end

function SamplesBuffer(init_value, buffersize, ll=nothing, empty=false)
    buffer = Array{eltype(init_value), 2}(undef, length(init_value)+(ll!==nothing), buffersize)

    if empty
        return SamplesBuffer(
            buffer,
            0,
            0,
            buffersize,
        )
    end

    if ll===nothing 
        buffer[:,1] .= init_value
    else
        buffer[1:end-1,1] .= init_value
        buffer[end,1] = ll
    end
    return SamplesBuffer(
        buffer,
        1,
        1,
        buffersize,
    )
end

function getlatestidx(buffer)
    if isbufferempty(buffer)
        return buffer.buffersize
    end
    return buffer.bufferidx
end

function viewlatest(buffer)
    latestidx = getlatestidx(buffer)
    return @view buffer.buffer[:, latestidx]
end

function addsample!(buffer, sample, ll=nothing)
    buffer.bufferidx += 1
    buffer.accepted_count += 1
    if ll===nothing
        buffer.buffer[:, buffer.bufferidx] .= sample
        latest = @view buffer.buffer[:, buffer.bufferidx]
    else
        buffer.buffer[1:end-1, buffer.bufferidx] .= sample
        latest = @view buffer.buffer[1:end-1, buffer.bufferidx]
        buffer.buffer[end, buffer.bufferidx] = ll
    end
    return latest
end

function repeatsample!(buffer)
    latestidx = getlatestidx(buffer)
    buffer.bufferidx += 1
    buffer.buffer[:, buffer.bufferidx] .= buffer.buffer[:, latestidx]
    latest = @view buffer.buffer[:, buffer.bufferidx]
    return latest
end

function isbufferempty(buffer)
    return buffer.bufferidx == 0
end

function isbufferfull(buffer)
    return buffer.bufferidx >= buffer.buffersize
end

function writebuffer!(io::IO, buffer, samples_count, thin)
    prev_buffer_last_n_skipped = ((thin-1)+samples_count - buffer.bufferidx)%thin
    first_ix = thin - prev_buffer_last_n_skipped
    @views write(io, buffer.buffer[:, first_ix:thin:buffer.bufferidx])
    buffer.bufferidx = 0
    buffer.accepted_count = 0
    return 
end

function writebuffer!(io::Base.DevNull, buffer, samples_count, thin)
    buffer.bufferidx = 0
    buffer.accepted_count = 0
    return 
end

function writebuffer!(out::SamplesBuffer, buffer, samples_count, thin)
    prev_buffer_last_n_skipped = ((thin-1)+samples_count - buffer.bufferidx)%thin
    first_ix = thin - prev_buffer_last_n_skipped
    for colix in first_ix:thin:buffer.bufferidx
        @views addsample!(out, buffer.buffer[:,colix])
    end
    buffer.bufferidx = 0
    buffer.accepted_count = 0
    return 
end

function writebuffer!(out::AbstractMatrix, buffer, samples_count, thin)
    prev_buffer_last_n_skipped = ((thin-1)+samples_count - buffer.bufferidx)%thin
    first_ix = thin - prev_buffer_last_n_skipped
    buffercols = first_ix:thin:buffer.bufferidx

    if samples_count <= buffer.bufferidx
        out_ix = 0
    else
        out_ix = (samples_count - buffer.bufferidx - 1)÷thin + 1
    end 

    for colix in buffercols
        out_ix += 1
        @views out[:, out_ix] .= buffer.buffer[:,colix]
    end
    buffer.bufferidx = 0
    buffer.accepted_count = 0
    return 
end