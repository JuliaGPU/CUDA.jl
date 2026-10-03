## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# On devices without concurrent managed access (Windows, Jetson boards up to Orin), the CPU
# cannot touch globally attached unified memory while any kernel is running, on any stream.
# Unified memory we allocate on such devices is therefore attached to the host, attached
# globally when it's used on the GPU, and attached to the host again before CPU access.
# Attaching to a single stream isn't an option: libraries like cuBLAS access memory from
# their own internal streams.
#
# This only covers allocations made by CUDA.jl. Attachment applies to an entire allocation,
# and a wrapper created with `unsafe_wrap` doesn't know the state of the allocation it points
# into, so it's never attached. Using a wrapper of host-attached memory on the GPU is the
# caller's responsibility (CUDA leaves that undefined).
@enum Attachment::UInt8 begin
  UNTRACKED           # not managed by us: other devices, pooled or wrapped memory
  HOST_ATTACHED
  GLOBALLY_ATTACHED
  ALWAYS_GLOBAL       # used by a graph, which can be launched at any time
end

function concurrent_managed_access(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    attribute(dev, DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) == 1
  end::Bool
end

# XXX: immutable with atomic refs?
mutable struct Managed{M}
  const mem::M
  const lock::ReentrantLock
  # which stream is currently using the memory, and in which context (which isn't known for
  # the default streams).
  stream::CuStream
  stream_ctx::CuContext
  generation::Int

  # the epoch of `stream` during which the memory was last used (see `StreamOrder`)
  epoch::UInt64

  # whether accessing this memory can cause implicit synchronization
  synchronizing::Bool

  # whether there are outstanding operations that haven't been synchronized
  dirty::Bool

  # whether the memory has been captured in a way that would make the dirty bit unreliable
  # (only for captures that CUDA.jl does not know about)
  captured::Bool

  # whether `stream` waits on the device for operations on other streams that used the
  # memory, which therefore haven't been synchronized either (implies `dirty`)
  waiting::Bool

  # whether a pointer was taken outside of an operation that CUDA.jl submits, and may thus
  # be used by work submitted to `stream` after it was stamped with `epoch`
  escaped::Bool

  # whether the memory was allocated by CUDA.jl, as opposed to imported using `unsafe_wrap`.
  # only such memory counts towards our memory usage, and can be reused once freed.
  const owned_allocation::Bool

  # how the visibility of unified memory is managed (see `Attachment`)
  attachment::Attachment

  # Graphs keep a lease beyond the lifetime of the array that owns this memory.
  Base.@atomic leases::Int
  Base.@atomic pending::Any

  function Managed(mem::AbstractMemory; stream = CUDACore.stream(), synchronizing = true,
                   dirty = true, captured = false, owned_allocation = false)
    # our unified allocations start out attached to the host (see `alloc_unified`)
    attachment = owned_allocation && mem isa UnifiedMemory && !mem.pooled &&
                 !concurrent_managed_access(device(mem.ctx)) ? HOST_ATTACHED : UNTRACKED
    # NOTE: memory starts as dirty, because stream-ordered allocations are only
    #       guaranteed to be physically allocated at a synchronization event.
    new{typeof(mem)}(mem, ReentrantLock(), stream, mem.ctx, generation(stream),
                     stream_epoch(stream), synchronizing, dirty, captured, false, false,
                     owned_allocation, attachment, 0, nothing)
  end
end

# Earlier generations have completed before the stream is handed to another task.
recycled(managed::Managed) = managed.generation != generation(managed.stream)

Base.sizeof(managed::Managed) = sizeof(managed.mem)

# the epoch to stamp an access on `stream` with. accesses on untracked streams are never
# covered by an event or synchronization.
function stream_epoch(stream::CuStream)
  order = stream.order
  order === nothing ? typemax(UInt64) : current_epoch(order)
end

# whether the last use of memory is known to have completed. memory whose pointer escaped
# may be used by work that was submitted after the epoch it was stamped with was closed.
function is_completed(managed::Managed)
  order = managed.stream.order
  !managed.captured && !managed.escaped && order !== nothing &&
    is_completed(order, managed.epoch)
end

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
    managed.waiting = false
  end
end

# an event to wait for the work on the stream that last used memory, or `nothing` if that
# work has finished. once the stream has been handed to another task (see `recycled`), it
# may be capturing the stream, so check that while holding the lock that recycling takes.
function pending_work(managed::Managed)
  @lock stream_disposal_lock begin
    recycled(managed) && return nothing
    stream = managed.stream
    order = stream.order
    order === nothing && return record_pending_work(stream)
    # a stream that `capture` is capturing on has an event that covers the work submitted
    # to it before, including the last use of the memory (captured operations don't take
    # ownership of memory). that's not enough for the capture itself, which may have
    # captured operations on the memory. the lock keeps a capture from beginning meanwhile.
    @lock order.lock begin
      event = order.capture_event
      if event === nothing || CUDACore.stream() == stream
        record_pending_work(stream)
      else
        event::CuEvent
      end
    end
  end
end
function record_pending_work(stream::CuStream)
  check_capture(stream)
  relaxed_capture_mode(() -> isdone(stream)) && return nothing
  event = CuEvent(EVENT_DISABLE_TIMING)
  cuEventRecord(event, stream)
  return event
end
function maybe_synchronize(managed::Managed)
  Base.@lock managed.lock begin
    if managed.synchronizing && (managed.dirty || managed.captured)
      if is_completed(managed)
        managed.dirty = false
        managed.waiting = false
      else
        synchronize(managed)
      end
    end
  end
end

# Attach memory globally before it's used on `stream`. Attaching is prohibited during stream
# capture, also on other streams unless the calling thread relaxes its capture mode, so for
# a capture the memory is attached on the allocation stream and kept global from then on.
function attach_globally!(managed::Managed{UnifiedMemory}, stream::CuStream, capturing::Bool)
  if managed.attachment == HOST_ATTACHED
    if capturing
      mem = managed.mem
      context!(mem.ctx) do
        side = allocation_stream(mem.ctx)
        # (synchronizing the allocation stream doesn't yield to other tasks)
        relaxed_capture_mode() do
          cuStreamAttachMemAsync(side, mem, 0, MEM_ATTACH_GLOBAL)
          cuStreamSynchronize(side)
        end
      end
    else
      cuStreamAttachMemAsync(stream, managed.mem, 0, MEM_ATTACH_GLOBAL)
    end
    managed.attachment = GLOBALLY_ATTACHED
  end
  if capturing && managed.attachment == GLOBALLY_ATTACHED
    managed.attachment = ALWAYS_GLOBAL
  end
  return
end

# Wait for the GPU to finish using memory before CPU access. Globally attached memory is
# also attached to the host again, ordered after its last use. That isn't done for memory
# with implicit synchronization disabled, or memory last used on one of the default streams
# (the per-thread one can't be targeted from another thread), so CPU access to those still
# requires the GPU to be idle. The caller holds `managed.lock`.
function prepare_host_access!(managed::Managed)
  if managed.attachment == GLOBALLY_ATTACHED && managed.synchronizing &&
     !managed.captured && managed.stream.ctx !== nothing
    event = @lock stream_disposal_lock begin
      # (holding the lock under which a capture on the stream begins, see `pending_work`)
      order = managed.stream.order
      order === nothing || lock(order.lock)
      try
        fence = order === nothing || recycled(managed) ? nothing : order.capture_event
        if fence !== nothing && CUDACore.stream() != managed.stream
          # the stream is being captured by another task, so attach on another stream,
          # after the work that was submitted before the capture began
          ctx = something(managed.stream.ctx)
          stream = disposal_stream(ctx)
        else
          fence = nothing
          if !recycled(managed) && isvalid(managed.stream)
            check_capture(managed.stream)
          end
          # this also handles owners that have been recycled or destroyed
          stream, ctx = release_stream(managed.stream, managed.stream_ctx, managed.generation)
        end
        context!(ctx) do
          fence === nothing || cuStreamWaitEvent(stream, fence::CuEvent, 0)
          # (attaching is prohibited while another thread captures in global mode)
          relaxed_capture_mode() do
            cuStreamAttachMemAsync(stream, managed.mem, 0, MEM_ATTACH_HOST)
          end
          event = CuEvent(EVENT_DISABLE_TIMING)
          cuEventRecord(event, stream)
          event
        end
      finally
        order === nothing || unlock(order.lock)
      end
    end
    synchronize(event)
    managed.attachment = HOST_ATTACHED
    managed.dirty = false
    managed.waiting = false
    check_exceptions()
  else
    maybe_synchronize(managed)
  end
  return
end

# Memory that moves to another stream needs to be ordered after the operations that used it
# before. For an operation that CUDA.jl submits to that stream itself, it suffices to make
# the stream wait for the previous one on the device. Other consumers, like a library that
# is passed a pointer, may access the memory from the host or from streams we don't know
# of, so for them the previous stream is synchronized from the host instead. Either way,
# this relies on the previous operations having been submitted already.

# whether memory can move from its stream to `stream` by waiting on the device. that needs
# two ordinary streams in the same context. memory used during unknown captures may also be
# used by graph launches we don't know of, and other kinds of memory can be accessed from the
# CPU in ways that we don't see. (known captures don't take ownership, see `take_ownership!`)
function can_handoff(managed::Managed{M}, stream::CuStream) where {M}
  source = managed.stream
  M == DeviceMemory && managed.synchronizing && !managed.captured &&
    source.order !== nothing && stream.order !== nothing && source.ctx == stream.ctx
end

# make `stream` wait on the device for the operations on the memory's stream, returning
# whether that was possible (see `stream_wait`)
function handoff!(managed::Managed, stream::CuStream)
  managed.dirty || return true
  if is_completed(managed)
    managed.dirty = false
    managed.waiting = false
    return true
  end
  source = managed.stream
  # (holding the lock that recycling streams takes, see `pending_work`)
  @lock stream_disposal_lock begin
    if recycled(managed)
      # the stream was handed to another task, which only happens when it was idle
      managed.dirty = false
      managed.waiting = false
      return true
    end
    isvalid(source) && stream_wait(stream, source) || return false
  end
  managed.waiting = true
  return true
end

# Transfer stream ownership of an allocation and mark it dirty in anticipation of an
# operation on it. The caller must hold `managed.lock` until that operation has been
# submitted, so the recorded owner cannot become visible before its submission. Pass
# `external=true` if the operation isn't submitted to `stream` by CUDA.jl (see above).
function take_ownership!(managed::Managed{M}; state=active_state(),
                         stream::CuStream=state.stream,
                         capturing::Bool=is_capturing(stream),
                         external::Bool=false) where {M}
  sizeof(managed) == 0 && return managed

  M == UnifiedMemory && attach_globally!(managed, stream, capturing)

  if capturing
    capture = current_capture(stream)
    if capture !== nothing
      # captured operations don't execute until the graph is launched, so only record the
      # use of the memory. the graph will take ownership of it when it is launched, which
      # orders the launch after earlier uses of the memory.
      record!(capture, managed)
      return managed
    end

    # an unknown capture: taint the memory so that we always synchronize
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
    if !(!external && can_handoff(managed, stream) && handoff!(managed, stream))
      maybe_synchronize(managed)
    end
    managed.stream = stream
    managed.stream_ctx = state.context
    # the work submitted to the previous stream has been waited for, which is all that's
    # known about uses of pointers that escaped (see `convert(::Type{CuPtr}, ::Managed)`)
    managed.escaped = false
  end
  managed.generation = generation(managed.stream)

  # an external consumer needs operations on other streams to have finished, also when the
  # stream already waits for them. operations on the stream itself are, as before, assumed
  # to be ordered by the consumer. (during an unknown capture, nothing is waited for)
  if external && managed.waiting && !capturing
    synchronize(managed)
  end

  # prefetch unified memory as we're likely to use it on the GPU
  if M == UnifiedMemory
    can_prefetch = !capturing
    can_prefetch &= !__pinned(convert(Ptr{Cvoid}, managed.mem), managed.mem.ctx)
    can_prefetch &= concurrent_managed_access(state.device)
    can_prefetch &= ndevices() == 1
    can_prefetch && prefetch(managed.mem; device=state.device, stream)
  end

  managed.epoch = stream_epoch(stream)
  managed.dirty = true
  return managed
end

function Base.convert(::Type{CuPtr{T}}, managed::Managed{M}) where {T,M}
  Base.@lock managed.lock begin
    # let null pointers pass through as-is
    ptr = convert(CuPtr{T}, managed.mem)
    ptr == CU_NULL && return ptr

    # within an operation that CUDA.jl submits (see `with_managed`), the pointer is used on
    # that operation's stream. otherwise, we don't know who is going to use it, or when.
    # it's assumed to be used on the task's stream, until the memory is used on another
    # stream: work submitted through the pointer after that isn't waited for. that work
    # may also be submitted after the stream was synchronized or an event was recorded,
    # so neither tells anything about the memory anymore (see `is_completed`).
    tls = task_local_state!()
    state = active_state(tls)
    stream = tls.operation_stream
    if stream === nothing
      take_ownership!(managed; state, stream=state.stream, external=true)
      managed.escaped = true
    else
      take_ownership!(managed; state, stream, capturing=tls.operation_capturing)
    end
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

    prepare_host_access!(managed)
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
