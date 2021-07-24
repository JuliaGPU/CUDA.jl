export SimpleNotificationPolicy, TreeNotificationPolicy

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
function Base.show(io::IO, ::TwoLevelPredictor)
    println(io, "TwoLevelPredictor")
end

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

function Base.show(io::IO, poller::VarPoller)
    print(io, "VarPoller($(poller.wait_durations), $(typeof(poller.branch_predictor)))")
end

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

into_time(x) = parse(Int64, x)

get_times() = map(into_time, split(readchomp(pipeline(`get_times.sh $(getpid())`, `tail -n $(Threads.nthreads())`))))

"""
    wait_and_kill_watcher(mod, poller::Poller, manager::AreaManager, policy::NotificationPolicy, event::CuEvent, ctx::CuContext,  pollers=1)

Called before executing device kernel.
This function starts an async task that polls according to the supplied `Poller`
while the device kernel runs.
"""
function wait_and_kill_watcher(mod::CuModule, poller::Poller, manager::AreaManager, policy::NoPolicy, event::CuEvent, pollers=1)
    return @async begin
        println("No policy specified")
    end
end

function wait_and_kill_watcher(mod::CuModule, poller::Poller, manager::AreaManager, policy::NotificationPolicy, event::CuEvent, pollers=1)
    t = reset_hostcall_area!(manager, policy, mod)
    if t === nothing
        return @async begin end
    end
    (area_ptr, meta_ptr, policy_ptr) = t
    f = (i) -> handle_hostcall(manager, area_ptr, meta_ptr, policy, policy_ptr, i)

    poll_intervals = poll_slices(policy, pollers)
    pollers = length(poll_intervals) # If asked for too many, this is reduced, so update value
    tasks = Task[]

    sem = Base.Semaphore(pollers)
    for i in 1:pollers # Flood semaphore
        Base.acquire(sem)
    end

    for (min, max) in poll_intervals
        t = @async begin
            Base.release(sem)

            yield()

            try
                # Hoist all these params
                if area_ptr !== nothing
                    launch_poller(poller, event, min, max, f)
                end
            catch e
                println("Failed $e")
                stacktrace()
            end
        end

        push!(tasks, t)
    end

    for i in 1:pollers
        Base.acquire(sem)
    end

    return @async begin
        start = get_times()[1]
        for i in 1:pollers
            Base.fetch(tasks[i])
        end
        end_time = get_times()[1]

        return (end_time - start)
    end
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

    i = min
    while true
        hostcalls = f(i)

        if isempty(hostcalls)
            do_poller(poller, 0)
        end

        for hostcall in hostcalls
            do_poller(poller, hostcall)
        end

        yield()
        i += 1

        if i > max
            i = min
            query(e) && break
        end
    end

    for i in min:max
        f(i)
    end
end


do_poller(poller::AlwaysPoller, hostcall::Int64) = ()
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
