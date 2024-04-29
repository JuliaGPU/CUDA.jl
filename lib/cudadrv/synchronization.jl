# support for nonblocking synchronization

const use_nonblocking_synchronization =
    Preferences.@load_preference("nonblocking_synchronization", true)


#
# bidirectional channel
#

# custom, unbuffered channel that supports returning a value to the sender
# without the need for a second channel
struct BidirectionalChannel{I,O} <: AbstractChannel{I}
    cond_take::Threads.Condition                 # waiting for data to become available
    cond_put::Threads.Condition                  # waiting for a writeable slot
    cond_ret::Threads.Condition                  # waiting for a data to be returned

    function BidirectionalChannel{I,O}() where {I,O}
        lock = ReentrantLock()
        cond_put = Threads.Condition(lock)
        cond_take = Threads.Condition(lock)
        cond_ret = Threads.Condition(lock)
        return new(cond_take, cond_put, cond_ret)
    end
end

Base.put!(c::BidirectionalChannel{I}, v) where {I} = put!(c, convert(I, v))
function Base.put!(c::BidirectionalChannel{I,O}, v::I) where {I,O}
    lock(c)
    try
        # wait for a slot to be available
        while isempty(c.cond_take)
            Base.wait(c.cond_put)
        end

        # pass a value to the consumer
        notify(c.cond_take, v, false, false)

        # wait for a return value to be produced
        Base.wait(c.cond_ret)::O
    finally
        unlock(c)
    end
end

function Base.take!(f::Base.Callable, c::BidirectionalChannel{I,O}) where {I,O}
    lock(c)
    try
        # notify the producer that we're ready to accept a value
        notify(c.cond_put, nothing, false, false)

        # receive a value from the producer
        v = Base.wait(c.cond_take)::I

        # return a value to the producer
        ret = f(v)::O
        notify(c.cond_ret, ret, false, false)
    finally
        unlock(c)
    end
end

Base.lock(c::BidirectionalChannel) = lock(c.cond_take)
Base.unlock(c::BidirectionalChannel) = unlock(c.cond_take)


#
# fast-path synchronization
#

# before using a nonblocking mechanism, which has some overhead, use a busy-loop
# that queries the state of the object to synchronize. this reduces latency,
# especially for short operations. note that because it does not actually perform
# the synchronization, when it returns true (indicating that the object is synchronized)
# the actual synchronization API should be called again.

function spinning_synchronization(f, obj)
    # fast path
    f(obj) && return true

    # minimize latency of short operations by busy-waiting,
    # initially without even yielding to other tasks
    spins = 0
    while spins < 256
        if spins < 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        f(obj) && return true
        spins += 1
    end

    return false
end


#
# nonblocking sync
#

@static if VERSION >= v"1.9.2"

# if we support foreign threads, perform the actual synchronization on a separate thread.

const SyncObject = Union{CuContext, CuStream, CuEvent}

const MAX_SYNC_THREADS = 4
const sync_channels = Array{BidirectionalChannel{SyncObject,CUresult}}(undef, MAX_SYNC_THREADS)
const sync_channel_cursor = Threads.Atomic{UInt32}(1)
const sync_channel_lock = Base.ReentrantLock()

function synchronization_worker(data)
    i = Int(data)
    chan = sync_channels[i]

    while true
        # wait for work
        take!(chan) do v
            if v isa CuContext
                context!(v)
                unchecked_cuCtxSynchronize()
            elseif v isa CuStream
                context!(v.ctx)
                unchecked_cuStreamSynchronize(v)
            elseif v isa CuEvent
                context!(v.ctx)
                unchecked_cuEventSynchronize(v)
            end
        end
    end
end

@noinline function create_synchronization_worker(i)
    lock(sync_channel_lock) do
        # test and test-and-set
        if isassigned(sync_channels, i)
            return
        end

        # should be safe to assign before threads are running;
        # any user will just submit work that makes it block
        sync_channels[i] = BidirectionalChannel{SyncObject,CUresult}()

        # we don't know what the size of uv_thread_t is, so reserve enough space
        tid = Ref{NTuple{32, UInt8}}(ntuple(i -> 0, 32))

        cb = @cfunction(synchronization_worker, Cvoid, (Ptr{Cvoid},))
        err = @ccall uv_thread_create(tid::Ptr{Cvoid}, cb::Ptr{Cvoid}, Ptr{Cvoid}(i)::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_create", err)
        @ccall uv_thread_detach(tid::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_detach", err)
    end

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
    res = put!(chan, val)
    # this `put!` blocks until the worker has finished processing and returned value
    # (which is different from regular channels)
    if res != SUCCESS
        throw_api_error(res)
    end

    return
end

function device_synchronize(; blocking::Bool=false, spin::Bool=true)
    if use_nonblocking_synchronization && !blocking
        if spin && spinning_synchronization(isdone, legacy_stream())
            cuCtxSynchronize()
        else
            maybe_collect(true)
            nonblocking_synchronize(context())
        end
    else
        maybe_collect(true)
        cuCtxSynchronize()
    end

    check_exceptions()
end

function synchronize(stream::CuStream=stream(); blocking::Bool=false, spin::Bool=true)
    if use_nonblocking_synchronization && !blocking
        if spin && spinning_synchronization(isdone, stream)
            cuStreamSynchronize(stream)
        else
            maybe_collect(true)
            nonblocking_synchronize(stream)
        end
    else
        maybe_collect(true)
        cuStreamSynchronize(stream)
    end

    check_exceptions()
end

function synchronize(event::CuEvent; blocking::Bool=false, spin::Bool=true)
    if use_nonblocking_synchronization && !blocking
        if spin && spinning_synchronization(isdone, event)
            cuEventSynchronize(event)
        else
            maybe_collect(true)
            nonblocking_synchronize(event)
        end
    else
        maybe_collect(true)
        cuEventSynchronize(event)
    end
end

else

# without thread adoption, have CUDA notify an async condition that wakes the libuv loop.
# this is not ideal: stream callbacks are deprecated, and do not fire in case of errors.
# furthermore, they do not trigger CUDA's synchronization hooks (see NVIDIA bug #3383169)
# requiring us to perform the actual API call again after nonblocking synchronization.

function nonblocking_synchronize(stream::CuStream)
    # wait for an event signalled by CUDA
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
                if isdone(stream)
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

function device_synchronize(; blocking::Bool=false, spin::Bool=true)
    if use_nonblocking_synchronization && !blocking
        stream = legacy_stream()
        if !spin || !spinning_synchronization(isdone, stream)
            nonblocking_synchronize(stream)
        end
    end
    maybe_collect(true)
    cuCtxSynchronize()

    check_exceptions()
end

function synchronize(stream::CuStream=stream(); blocking::Bool=false, spin::Bool=true)
    if use_nonblocking_synchronization && !blocking
        if !spin || !spinning_synchronization(isdone, stream)
            nonblocking_synchronize(stream)
        end
    end
    maybe_collect(true)
    cuStreamSynchronize(stream)

    check_exceptions()
end

function synchronize(event::CuEvent; blocking::Bool=false, spin::Bool=true)
    if use_nonblocking_synchronization && !blocking
        spin && spinning_synchronization(isdone, event)
    end
    maybe_collect(true)
    cuEventSynchronize(event)
end

end
