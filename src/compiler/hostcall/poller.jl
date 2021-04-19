# Julia sleep only works for ms
usleep(usecs) = ccall(:usleep, Cint, (Cuint,), usecs)

abstract type Poller end

struct AlwaysPoller <: Poller
    count :: Int32
end
struct ConstantPoller <: Poller
    dur :: UInt32
end
mutable struct VarPoller <: Poller

end


"""
    wait_and_kill_watcher(poller::P, event::CuEvent, ctx::CuContext)

Called before executing device kernel.
This function starts an async task that polls according to the supplied `Poller`
while the device kernel runs.
"""
function wait_and_kill_watcher(mod::CuModule, poller::P, manager::AreaManager, event::CuEvent, ) where {P<:Poller}
    # empty!(host_refs)
    reset()
    area = reset_hostcall_area!(manager, mod)

    t = @async begin
        yield()

        try
            launch_poller(poller, manager, event, area)
        catch e
            println("Failed $e")
            stacktrace()
        end

        println("$(val())")
    end

    while !istaskstarted(t)
        yield()
    end

    return t
end

"""
Polls as often as is allowed.
"""
function launch_poller(poller::AlwaysPoller, manager::AreaManager, e::CuEvent, area::Ptr{Int64})
    count = area_count(manager)

    while !query(e)
        for i in 1:count
            handle_hostcall(manager, area, i)
        end
        yield()
    end

    for i in 1:count
        handle_hostcall(manager, area, i)
    end
end


"""
Polls all hostcall areas then sleeps for a certain duration.
"""
function launch_poller(poller::ConstantPoller, manager::AreaManager, e::CuEvent, area::Ptr{Int64})
    count = area_count(manager)

    while !query(e)
        for i in 1:count
            handle_hostcall(manager, area, i)
        end

        usleep(poller.dur)
    end

    for i in 1:count
        handle_hostcall(manager, area, i)
    end
end


function launch_poller(poller::VarPoller, e::CuEvent, ctx::CuContext)
    error("Not yet supported")
end
