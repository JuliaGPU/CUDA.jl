using Printf

# Julia sleep only works for ms
usleep(usecs) = ccall(:usleep, Cint, (Cuint,), usecs)

abstract type Poller end
abstract type WaitPredictor end

# Maybe: this is just a primitive with bounds qq
mutable struct SaturationCounter <: WaitPredictor
    state::Int64
    max_state::Int64
end
SaturationCounter(max::Int64) = SaturationCounter(div(max, 2), max)

mutable struct TwoLevelPredictor <: WaitPredictor
    counters::Vector{SaturationCounter}
    history::UInt8
end
TwoLevelPredictor(counter::SaturationCounter) = TwoLevelPredictor([deepcopy(counter) for i in 1:256], 0)


struct AlwaysPoller <: Poller
    count :: Int32
end

struct ConstantPoller <: Poller
    dur :: UInt32
end

struct VarPoller <: Poller
    branch_predictor::WaitPredictor
    wait_durations::Vector{Int64}
end
VarPoller(wait_durations::Vector{Int64}, ::Type{SaturationCounter}) = VarPoller(SaturationCounter(length(wait_durations)), wait_durations)
VarPoller(wait_durations::Vector{Int64}, ::Type{TwoLevelPredictor}) = VarPoller(TwoLevelPredictor(SaturationCounter(length(wait_durations))), wait_durations)

current(predictor::SaturationCounter) = predictor.state
function correct!(predictor::SaturationCounter, value::Int64)
    predictor.state += value == 0 ? -1 : 1
    predictor.state <= 0 && (predictor.state = 1)
    predictor.state > predictor.max_state && (predictor.state = predictor.max_state)
end

current(predictor::TwoLevelPredictor) = current(@inbounds predictor.counters[predictor.history + 1])
function correct!(predictor::TwoLevelPredictor, value::Int64)
    @inbounds correct!(predictor.counters[predictor.history + 1], value)

    if value != 0
        predictor.history = predictor.history << 2 | UInt8(value % 256)
    end
end


"""
    wait_and_kill_watcher(poller::P, event::CuEvent, ctx::CuContext)

Called before executing device kernel.
This function starts an async task that polls according to the supplied `Poller`
while the device kernel runs.
"""
function wait_and_kill_watcher(mod::CuModule, poller::Poller, manager::AreaManager, policy::NotificationPolicy, event::CuEvent)
    (area_ptr, policy_ptr) = reset_hostcall_area!(manager, policy, mod)
    f = (i) -> handle_hostcall(manager, area_ptr, policy, policy_ptr, i)

    t = @async begin
        yield()
        # This should be gotten from the policy
        count = area_count(manager)

        try
            # Hoist all these params
            area_ptr !== nothing && launch_poller(poller, event, 1, count, f)
        catch e
            println("Failed $e")
            stacktrace()
        end

        println(policy)

        # println("$(val())")
    end

    while !istaskstarted(t)
        yield()
    end

    return t
end


function print_stats(values)
    total = sum(values)
    mean = Statistics.mean(values)
    median = Statistics.median(values)
    var = Statistics.var(values)
    minv = min(values...)
    maxv = max(values...)

    @printf "total %.6fs, mean %.6fs, median %.6fs, var %.6fs, min %.6fs, max %.6fs" total mean median var minv maxv
end


"""
Polls all hostcall areas then sleeps for a certain duration.
"""
function launch_poller(poller::Poller, e::CuEvent, min::Int64, max::Int64, f::Function)
    hits = 0
    misses = 0
    hit_times = Float64[]
    miss_times = Float64[]

    i = min
    while true
        s_time = time()

        hostcalls = f(i)

        if isempty(hostcalls)
            misses += 1
            push!(miss_times, time() - s_time)
            do_poller(poller, 0)
        else
            hits += 1
            push!(hit_times, time() - s_time)
        end

        for hostcall in hostcalls
            do_poller(poller, hostcall)
        end

        i += 1

        if i > max
            i = min
            query(e) && break
        end
    end

    for i in min:max
        f(i)
    end

    @printf "hits %d, misses %d\n" hits misses
    print("Hit  stats: "); print_stats(hit_times); println()
    print("MIss stats: "); print_stats(miss_times); println()
end


do_poller(poller::AlwaysPoller, hostcall::Int64) = yield()
do_poller(poller::ConstantPoller, hostcall::Int64) = hostcall == 0 && (usleep(poller.dur))
function do_poller(poller::VarPoller, hostcall::Int64)
    sleep = current(poller.branch_predictor)
    if hostcall == 0
        usleep(sleep)
    end
    correct!(poller.branch_predictor, hostcall)
end

function launch_poller(poller::VarPoller, e::CuEvent, ctx::CuContext)
    error("Not yet supported")
end

# AlwaysPoller
#     zonder gather
#         hits 9174, misses 67626
#         0.169635 seconds
#     met gather
#         hits 4059, misses 22261
#         0.065526 seconds

# ConstantPoller(50)
#     zonder gather
#         hits 9174, misses 41386
#         4.275604 seconds
#     met gather
#         hits 4059, misses 2901
#         0.307719 seconds

# VarPoller(CUDA.SaturationCounter(5), [5, 50, 100, 150, 200])
#     zonder gather
#         hits 9174, misses 44586
#         2.542565 seconds
#     met gather
#         hits 4059, misses 3541
#         0.243481 seconds
