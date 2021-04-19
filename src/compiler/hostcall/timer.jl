using Printf
using Statistics
# volledige time
# time per syscall

# time / area count

# periode tussen 2 hostcalls
# totale gebruikte tijd poller


mutable struct Timer
    last_start::Union{Nothing,Float64}
    vals::Vector{Float64}
end
Timer() = Timer(nothing, Float64[])

mutable struct Counter
    counts::Dict{String}{Int64}
    times::Dict{String}{Timer}
end

Counter() = Counter(Dict(), Dict())
global_counter = Counter()

# inc(x...) = ()
# start(x...) = ()
# stop(x...) = ()


val() = global_counter
reset() = (global_counter.counts = Dict(); global_counter.times = Dict())

function inc(x::String, v=1)
    haskey(global_counter.counts, x) || (global_counter.counts[x] = 0)
    global_counter.counts[x] += v
end

start(x::String) = (get!(global_counter.times, x, Timer()).last_start = time())
function stop(x::String, y...)
    haskey(global_counter.times, x) || error("Cannot stop what isn't present $x")
    timerx = global_counter.times[x]
    timerx.last_start === nothing && error("Cannot stop what isn't started $x")

    t = time()
    delta = t - timerx.last_start
    timerx.last_start = nothing
    push!(timerx.vals, delta)

    for i in y
        t = get!(global_counter.times, i, Timer())
        push!(t.vals, delta)
    end
end

function Base.show(io::IO, u::Timer)
    total = sum(u.vals)
    count = length(u.vals)
    mean = Statistics.mean(u.vals)
    median = Statistics.median(u.vals)
    var = Statistics.var(u.vals)
    minv = min(u.vals...)
    maxv = max(u.vals...)

    @printf(io, "count %-5d total %.6fs, mean %.6fs, median %.6fs, var %.6fs, min %.6fs, max %.6fs", count, total, mean, median, var, minv, maxv)
end

function Base.show(io::IO, u::Counter)
    println(io, "\nPlain counts")
    for (key, value) in u.counts
        @printf(io, "%-26s", key); print(io, ": "); println(io, value)
    end

    println(io, "\nTimes")
    for (key, value) in u.times
        isempty(value.vals) && continue
        @printf(io, "%-26s", key); print(io, ": "); println(io, value)
    end
end
