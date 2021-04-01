# Julia sleep only works for ms
usleep(usecs) = ccall(:usleep, Cint, (Cuint,), usecs)

abstract type Poller end

struct AlwaysPoller <: Poller end
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
function wait_and_kill_watcher(poller::P, event::CuEvent, ctx::CuContext) where {P<:Poller}
    empty!(host_refs)
    t = @async begin
        has_hostcalls(ctx) || return
        # reset!(timer)

        println("Start the watcher!")
        try
            launch_poller(poller, event, ctx)
        catch e
            println("Failed $e")
            stacktrace()
        end
        println("Killed the watcher!")
    end

    while !istaskstarted(t)
        yield()
    end

    return t
end

"""
Polls as often as is allowed.
"""
function launch_poller(::AlwaysPoller, e::CuEvent, ctx::CuContext)
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, hostcall_areas[ctx].ptr)
    count = hostcall_area_count()-1
    size = hostcall_area_size()

    while !query(e)
        for i in 0:count
            if handle_hostcall(llvmptr + i * size)
                println("handled $i")
            end
        end
        yield()
    end
end


"""
Polls all hostcall areas then sleeps for a certain duration.
"""
function launch_poller(poller::ConstantPoller, e::CuEvent, ctx::CuContext)
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, hostcall_areas[ctx].ptr)
    count = hostcall_area_count()-1
    size = hostcall_area_size()

    while !query(e)
        for i in 0:count
            handle_hostcall(llvmptr + i * size)
        end

        usleep(poller.dur)
    end
end


function launch_poller(poller::VarPoller, e::CuEvent, ctx::CuContext)
    error("Not yet supported")
end
