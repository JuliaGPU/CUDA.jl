"""
All Timer related things are only used during devolping to help gauge performance.

It counts usefull and useless calls to `handle_hostcall`, in plain count and execution time.
"""

mutable struct Timer
    sample::Int64
    total_duration::Float64
    total_duration_useless::Float64
    count::Int64
    count_useless::Int64
    last_start::Union{Nothing,Float64}
end

function start_sample!(timer::Timer)
    timer.sample += 1
end

const timer = Timer(0,0,0,0,0,nothing)

function start!(timer::Timer)
    sample = time()
    timer.last_start = sample
end


function stop!(timer::Timer, useless=True)
    sample = time()
    if isa(timer.last_start, Float64)
        delta = sample - timer.last_start
        if useless
            timer.total_duration_useless += delta
            timer.count_useless += 1
        end
        timer.total_duration += delta

        timer.count += 1
    else
        println("Something fucked")
    end
    timer.last_start = nothing
end

function Base.show(io::IO, timer::Timer)
    percentage_time = timer.total_duration_useless/timer.total_duration * 100
    percentage_count = Float32(timer.count_useless) / Float32(timer.count) * 100
    @printf(io, "<Timer tried: %d (%.3f%% useless) in %.3fs (%.3f%% useless) samples=%d>",
        timer.count / timer.sample, percentage_count,timer.total_duration / timer.sample, percentage_time, timer.sample)
end

function reset!(timer::Timer)
    timer.last_start = nothing
    timer.count = 0
    timer.sample = 0
    timer.count_useless = 0
    timer.total_duration = 0
    timer.total_duration_useless = 0
end

function time_it(f::Function)
    reset!(timer)

    f(timer)

    return timer
end
