# support for nonblocking synchronization

#
# bidirectional channel
#

# custom, unbuffered channel that supports returning a value to the sender
# without the need for a second channel
struct BidirectionalChannel{T} <: AbstractChannel{T}
    cond_take::Threads.Condition                 # waiting for data to become available
    cond_put::Threads.Condition                  # waiting for a writeable slot
    cond_ret::Threads.Condition                  # waiting for a data to be returned

    function BidirectionalChannel{T}() where T
        lock = ReentrantLock()
        cond_put = Threads.Condition(lock)
        cond_take = Threads.Condition(lock)
        cond_ret = Threads.Condition(lock)
        return new(cond_take, cond_put, cond_ret)
    end
end

Base.put!(c::BidirectionalChannel{T}, v) where T = put!(c, convert(T, v))
function Base.put!(c::BidirectionalChannel{T}, v::T) where T
    lock(c)
    try
        # wait for a slot to be available
        while isempty(c.cond_take)
            Base.wait(c.cond_put)
        end

        # pass a value to the consumer
        notify(c.cond_take, v, false, false)

        # wait for a return value to be produced
        Base.wait(c.cond_ret)
    finally
        unlock(c)
    end
end

function Base.take!(f, c::BidirectionalChannel{T}) where T
    lock(c)
    try
        # notify the producer that we're ready to accept a value
        notify(c.cond_put, nothing, false, false)

        # receive a value from the producer
        v = Base.wait(c.cond_take)::T

        # return a value to the producer
        ret = f(v)
        notify(c.cond_ret, ret, false, false)
    finally
        unlock(c)
    end
end

Base.lock(c::BidirectionalChannel) = lock(c.cond_take)
Base.unlock(c::BidirectionalChannel) = unlock(c.cond_take)


#
# nonblocking sync
#

if VERSION >= v"1.9.3"

# if we support foreign threads, perform the synchronization on a separate thread.

const MAX_SYNC_THREADS = 4
const sync_channels = Array{BidirectionalChannel{Any}}(undef, MAX_SYNC_THREADS)
const sync_channel_cursor = Threads.Atomic{UInt32}(1)

function synchronization_worker(data)
    i = Int(data)
    chan = sync_channels[i]

    while true
        # wait for work
        lock(chan)
        try
            take!(chan) do v
                # TODO: don't use `context!`, but use raw API calls that don't require TLS
                if v isa CuContext
                    context!(v)
                    unsafe_cuCtxSynchronize()
                elseif v isa CuStream
                    context!(v.ctx)
                    unsafe_cuStreamSynchronize(v)
                elseif v isa CuEvent
                    context!(v.ctx)
                    unsafe_cuEventSynchronize(v)
                end
            end
        finally
            unlock(chan)
        end
    end
end

@noinline function create_synchronization_worker(i)
    sync_channels[i] = BidirectionalChannel{Any}()
    # should be safe to assign before threads are running;
    #  any user will just submit work that makes it block

    # we don't know what the size of uv_thread_t is, so reserve enough space
    tid = Ref{NTuple{32, UInt8}}(ntuple(i -> 0, 32))

    cb = @cfunction(synchronization_worker, Cvoid, (Ptr{Cvoid},))
    @ccall uv_thread_create(tid::Ptr{Cvoid}, cb::Ptr{Cvoid}, Ptr{Cvoid}(i)::Ptr{Cvoid})::Int32

    return
end

function nonblocking_synchronize(val)
    # get the channel of a synchronization worker
    i = mod1(Threads.atomic_add!(sync_channel_cursor, UInt32(1)), MAX_SYNC_THREADS)
    if !isassigned(sync_channels, i)
        # TODO: write lock, double check, etc
        create_synchronization_worker(i)
    end
    chan = @inbounds sync_channels[i]

    # submit the object to synchronize
    lock(chan)
    try
        res = put!(chan, val)
        # this `put!` blocks until the worker has finished processing and returned value
        # (which is different from regular channels)
        if res != SUCCESS
            throw_api_error(res)
        end
    finally
        unlock(chan)
    end

    return
end

@inline function device_synchronize()
    nonblocking_synchronize(context())
    check_exceptions()
end

function synchronize(stream::CuStream=stream())
    # fast path
    isdone(stream) && return

    nonblocking_synchronize(stream)
    check_exceptions()
end

function synchronize(event::CuEvent)
    # fast path
    isdone(event) && return

    nonblocking_synchronize(event)
end

else

# without thread adoption, have CUDA notify an async condition that wakes the libuv loop.
# this is not ideal: stream callbacks are deprecated, and do not fire in case of errors.
# furthermore, they do not trigger CUDA's synchronization hooks (see NVIDIA bug #3383169)
# requiring us to perform the actual API call again after nonblocking synchronization.

@inline function nonblocking_synchronize(stream::CuStream)
    # fast path
    isdone(stream) && return

    # minimize latency of short operations by busy-waiting,
    # initially without even yielding to other tasks
    spins = 0
    while spins < 256
        if spins < 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # Temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        isdone(stream) && return
        spins += 1
    end

    # minimize CPU usage of long-running kernels by waiting for an event signalled by CUDA
    event = Base.Event()
    launch(; stream) do
        notify(event)
    end
    # if an error occurs, the callback may never fire, so use a timer to detect such cases
    dev = device()
    timer = Timer(0; interval=1)
    Base.@sync begin
        Threads.@spawn try
            device!(dev)
            while true
                try
                    Base.wait(timer)
                catch err
                    err isa EOFError && break
                    rethrow()
                end
                if unsafe_cuStreamQuery(stream) != ERROR_NOT_READY
                    break
                end
            end
        finally
            notify(event)
        end

        Threads.@spawn begin
            Base.wait(event)
            close(timer)
        end
    end

    return
end

@inline function device_synchronize()
    nonblocking_synchronize(legacy_stream())
    cuCtxSynchronize()

    check_exceptions()
end

function synchronize(stream::CuStream=stream())
    nonblocking_synchronize(stream)
    cuStreamSynchronize(stream)

    check_exceptions()
end

function synchronize(e::CuEvent)
    # fast path
    isdone(e) && return

    # spin (initially without yielding to minimize latency)
    spins = 0
    while spins < 256
        if spins < 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # Temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        isdone(e) && return
        spins += 1
    end

    cuEventSynchronize(e)

    return
end

end
