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
    empty!(host_refs)
    reset_hostcall_area!(manager, mod)
    ctx = mod.ctx

    t = @async begin
        yield()

        try
            launch_poller(poller, manager, event, ctx)
        catch e
            println("Failed $e")
            stacktrace()
        end
    end

    while !istaskstarted(t)
        yield()
    end

    return t
end

"""
Polls as often as is allowed.
"""
function launch_poller(poller::AlwaysPoller, manager::AreaManager, e::CuEvent, ctx::CuContext)
    count = area_count(manager)

    while !query(e)
        for i in 1:count
            handle_hostcall(manager, ctx, i)
        end
        yield()
    end
end


"""
Polls all hostcall areas then sleeps for a certain duration.
"""
function launch_poller(poller::ConstantPoller, manager::AreaManager, e::CuEvent, ctx::CuContext)
    count = area_count(manager)

    while !query(e)
        for i in 1:count
            handle_hostcall(manager, ctx, i)
        end

        usleep(poller.dur)
    end
end


function launch_poller(poller::VarPoller, e::CuEvent, ctx::CuContext)
    error("Not yet supported")
end
