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
    vals::Vector{Int64}
end
Counter() = Counter(Int64[])

mutable struct Stats
    counts::Dict{String}{Counter}
    times::Dict{String}{Timer}
end

Stats() = Stats(Dict(), Dict())
global_counter = Stats()

inc(x...) = ()
start(x...) = ()
stop(x...) = ()


val(;counter=global_counter) = counter
reset(;counter=global_counter) = (counter.counts = Dict(); counter.times = Dict())

# function inc(x::String, v=1;counter=global_counter)
#     haskey(counter.counts, x) || (counter.counts[x] = Counter())
#     push!(counter.counts[x].vals, v)
# end

# start(x::String;counter=global_counter) = (get!(counter.times, x, Timer()).last_start = time())
# function stop(x::String, y...; counter=global_counter)
#     haskey(counter.times, x) || error("Cannot stop what isn't present $x")
#     timerx = counter.times[x]
#     timerx.last_start === nothing && error("Cannot stop what isn't started $x")

#     t = time()
#     delta = t - timerx.last_start
#     timerx.last_start = nothing
#     push!(timerx.vals, delta)

#     for i in y
#         t = get!(counter.times, i, Timer())
#         push!(t.vals, delta)
#     end
# end

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
    total = sum(u.vals)
    mean = Statistics.mean(u.vals)
    median = Statistics.median(u.vals)
    var = Statistics.var(u.vals)
    minv = min(u.vals...)
    maxv = max(u.vals...)

    @printf(io, "total %-5d mean %.6fs, median %.6fs, var %.6fs, min %.6fs, max %.6fs", total, mean, median, var, minv, maxv)
end

function Base.show(io::IO, u::Stats)
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
