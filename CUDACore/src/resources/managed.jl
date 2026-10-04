## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# XXX: immutable with atomic refs?
mutable struct Managed{M}
  const mem::M
  const lock::ReentrantLock
  # which stream is currently using the memory, and in which context (which isn't known for
  # the default streams).
  stream::CuStream
  stream_ctx::CuContext
  generation::Int

  # whether accessing this memory can cause implicit synchronization
  synchronizing::Bool

  # whether there are outstanding operations that haven't been synchronized
  dirty::Bool

  # whether the memory has been captured in a way that would make the dirty bit unreliable
  captured::Bool

  # whether the memory was allocated by CUDA.jl, as opposed to imported using `unsafe_wrap`.
  # only such memory counts towards our memory usage, and can be reused once freed.
  const owned_allocation::Bool

  # Graphs keep a lease beyond the lifetime of the array that owns this memory.
  Base.@atomic leases::Int
  Base.@atomic pending::Any

  function Managed(mem::AbstractMemory; stream = CUDACore.stream(), synchronizing = true,
                   dirty = true, captured = false, owned_allocation = false)
    # NOTE: memory starts as dirty, because stream-ordered allocations are only
    #       guaranteed to be physically allocated at a synchronization event.
    new{typeof(mem)}(mem, ReentrantLock(), stream, mem.ctx, generation(stream),
                     synchronizing, dirty, captured, owned_allocation, 0, nothing)
  end
end

# Earlier generations have completed before the stream is handed to another task.
recycled(managed::Managed) = managed.generation != generation(managed.stream)

Base.sizeof(managed::Managed) = sizeof(managed.mem)

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
  Base.@lock managed.lock begin
    # the default streams need to be synchronized in the context they were used in
    context!(managed.stream_ctx) do
      if on_per_thread_stream(managed.stream)
        # that one is specific to the thread that used it, which isn't known
        device_synchronize()
      else
        event = pending_work(managed)
        event === nothing || synchronize(event)
        # (the work may have raised an exception)
        check_exceptions()
      end
    end
    managed.dirty = false
  end
end

# an event to wait for the work on the stream that last used memory, or `nothing` if that
# work has finished. once the stream has been handed to another task (see `recycled`), it
# may be capturing the stream, so check that while holding the lock that recycling takes.
function pending_work(managed::Managed)
  @lock stream_disposal_lock begin
    recycled(managed) && return nothing
    relaxed_capture_mode(() -> isdone(managed.stream)) && return nothing
    event = CuEvent(EVENT_DISABLE_TIMING)
    record(event, managed.stream)
    return event
  end
end
function maybe_synchronize(managed::Managed)
  Base.@lock managed.lock begin
    if managed.synchronizing && (managed.dirty || managed.captured)
      synchronize(managed)
    end
  end
end

# Transfer stream ownership of an allocation and mark it dirty in anticipation of a
# device-side operation. The caller must hold `managed.lock` until that operation has been
# submitted to `stream`, so the recorded owner cannot become visible before its submission.
function take_ownership!(managed::Managed{M}; state=active_state(),
                         stream::CuStream=state.stream,
                         capturing::Bool=is_capturing(stream)) where {M}
  sizeof(managed) == 0 && return managed

  # accessing memory during stream capture: taint the memory so that we always synchronize
  if capturing
    managed.captured = true
  end

  # accessing memory on another device: ensure the data is ready and accessible
  if M == DeviceMemory && state.context != managed.mem.ctx
    maybe_synchronize(managed)
    source_device = managed.mem.dev

    # enable peer-to-peer access
    if maybe_enable_peer_access(state.device, source_device) != 1
        throw(ArgumentError(
            """cannot take the GPU address of inaccessible device memory.

               You are trying to use memory from GPU $(deviceid(source_device)) on GPU $(deviceid(state.device)).
               P2P access between these devices is not possible; either switch to GPU $(deviceid(source_device))
               by calling `CUDA.device!($(deviceid(source_device)))`, or copy the data to an array allocated on device $(deviceid(state.device))."""))
    end

    # set pool visibility
    # XXX: disabled because of NVIDIA bug #6098762
    #if stream_ordered(source_device)
    #  pool = pool_create(source_device)
    #  access!(pool, state.device, ACCESS_FLAGS_PROT_READWRITE)
    #end
  end

  # accessing memory on another stream: ensure the data is ready and take ownership.
  # (the default streams are specific to a context, so also check that.)
  if managed.stream != stream || managed.stream_ctx != state.context
    maybe_synchronize(managed)
    managed.stream = stream
    managed.stream_ctx = state.context
  end
  managed.generation = generation(managed.stream)

  # prefetch unified memory as we're likely to use it on the GPU
  if M == UnifiedMemory
    can_prefetch = !capturing
    can_prefetch &= !__pinned(convert(Ptr{Cvoid}, managed.mem), managed.mem.ctx)
    can_prefetch &= attribute(state.device,
                              DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) == 1
    can_prefetch &= ndevices() == 1
    can_prefetch && prefetch(managed.mem; device=state.device, stream)
  end

  managed.dirty = true
  return managed
end

function Base.convert(::Type{CuPtr{T}}, managed::Managed{M}) where {T,M}
  Base.@lock managed.lock begin
    # let null pointers pass through as-is
    ptr = convert(CuPtr{T}, managed.mem)
    ptr == CU_NULL && return ptr

    state = active_state()
    take_ownership!(managed; state, stream=state.stream)
    return ptr
  end
end

function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T,M}
  Base.@lock managed.lock begin
    # let null pointers pass through as-is
    ptr = convert(Ptr{T}, managed.mem)
    ptr == C_NULL && return ptr

    # accessing memory on the CPU: only allowed for host or unified allocations
    if M == DeviceMemory
      throw(ArgumentError(
          """cannot take the CPU address of GPU memory.

             You are probably falling back to or otherwise calling CPU functionality
             with GPU array inputs. This is not supported by regular device memory;
             ensure this operation is supported by CUDA.jl, and if it isn't, try to
             avoid it or rephrase it in terms of supported operations. Alternatively,
             you can consider using GPU arrays backed by unified memory by
             allocating using `cu(...; unified=true)`."""))
    end

    # make sure any work on the memory has finished.
    maybe_synchronize(managed)
    return ptr
  end
end


## leases
#
# graphs use memory whenever they are launched, long after the operations that use the memory
# were captured. to keep that memory alive, graphs lease it: releasing leased memory (from
# whatever owns it, e.g., an array that's been freed) is postponed until the last lease ends.
# all ways to release managed memory go through `release`, so that they respect leases.

"""
    lease!(managed::Managed)

Prevent memory from being released until a matching call to [`unlease!`](@ref).
"""
function lease!(managed::Managed)
  Base.@atomic managed.leases += 1
  return managed
end

"""
    unlease!(managed::Managed)

End a lease on memory, releasing it if it was released while leased.
"""
function unlease!(managed::Managed)
  if (Base.@atomic managed.leases -= 1) == 0
    f = Base.@atomicswap managed.pending = nothing
    f === nothing || f(managed)
  end
  return
end

# release memory by calling `f(managed)`, now, or when the last lease ends. this can be
# called from a finalizer, as long as `f` can be.
function release(f, managed::Managed)
  if (Base.@atomic managed.leases) > 0
    Base.@atomic managed.pending = f
    # the last lease may have ended in the meantime, in which case `unlease!` might not have
    # seen `f`. whoever takes it from `pending` first calls it.
    (Base.@atomic managed.leases) > 0 && return
    f = Base.@atomicswap managed.pending = nothing
    f === nothing && return
  end
  f(managed)
  return
end
