## registered host memory
#
# unregistering host memory with `cuMemHostUnregister` waits for all running kernels to
# finish, also blocking kernel launches from other threads in the meantime. that is deferred
# until reclaiming memory, or until many registrations are pending, keeping the owner of the
# memory alive in the meantime so that it cannot be reused while still registered.

struct RetiredRegistration
  mem::HostMemory
  # registered by `__pin`, which counts registrations
  counted::Bool
  # what keeps the memory alive (if anything)
  owner::Any
end

const pending_registration_bytes = Threads.Atomic{Int}(0)

function release_now(reg::RetiredRegistration)
  Threads.atomic_add!(pending_registration_bytes, sizeof(reg.mem))
  hold!(ReleaseAction(unregister_retired!, reg, reg.mem.ctx, true))
end
release_registration(reg::RetiredRegistration) = discard(reg)

function unregister_retired!(reg::RetiredRegistration)
  if reg.counted
    __unpin(pointer(reg.mem), reg.mem.ctx)
  else
    unregister(reg.mem)
  end
  Threads.atomic_sub!(pending_registration_bytes, sizeof(reg.mem))
  return
end

# this waits for the GPU, so it shouldn't be called while holding a lock
function unregister_pending!()
  releasing(; blocking=true) do
    drain_retired()
    release_held!(; reclaim=true, select=action -> action.obj isa RetiredRegistration)
  end
end

# before registering more memory, unregister pending registrations if there are many
function maybe_unregister_pending!()
  drain_retired(ALLOC_DRAIN_LIMIT)
  if pending_registration_bytes[] > Sys.total_memory() ÷ 20
    unregister_pending!()
  end
  return
end

# register memory, or return nothing if it is still registered, e.g., by a registration
# that is pending, when the memory of an array that was wrapped by pointer has been reused
function try_register(ptr::Ptr, sz::Integer, flags=0)
  try
    register(HostMemory, ptr, sz, flags)
  catch err
    (err isa CuError && err.code == ERROR_HOST_MEMORY_ALREADY_REGISTERED) || rethrow()
    nothing
  end
end

function register_host_memory(ptr::Ptr, sz::Integer, flags=0)
  maybe_unregister_pending!()
  mem = try_register(ptr, sz, flags)
  if mem === nothing
    unregister_pending!()
    mem = register(HostMemory, ptr, sz, flags)
  end
  return mem
end


## memory pinning

@public pin

# protects the registries below. as unregistering memory waits for the GPU, that's done
# without holding this lock.
const pin_lock = ReentrantLock()

struct PinnedObject
    ref::WeakRef
    size::Int  # memory size in bytes
end

# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_objects = Dict{Tuple{CuContext,Ptr{Cvoid}}, PinnedObject}()

"""
    pin(a::AbstractArray)
    pin(ref::Base.RefValue)

Page-lock (pin) the host memory backing `a`, which makes copies between it and the GPU
faster, and makes it possible to perform them asynchronously.

The memory stays pinned for the lifetime of `a`. Unpinning memory waits for all running
kernels to finish, so once `a` has been garbage collected, its memory is only unpinned (and
freed) when memory is reclaimed, e.g., when calling `CUDA.reclaim()`, or when a lot of
memory is waiting to be unpinned.
"""
function pin(a::AbstractArray)
    ctx = context()
    ptr = pointer(a)
    key = (ctx, convert(Ptr{Nothing}, ptr))

    # only pin an object once per context
    previous, pinned = Base.@lock pin_lock begin
        get(__pinned_objects, key, nothing), haskey(__pin_count, key)
    end
    already_owned = previous !== nothing && previous.ref.value !== nothing
    if already_owned && pinned
        sizeof(a) == previous.size && return nothing
        # the object was resized: replace its registration, but not its finalizer
        __unpin(ptr, ctx)
    end
    __pin(ptr, sizeof(a))
    Base.@lock pin_lock begin
        __pinned_objects[key] = PinnedObject(WeakRef(a), sizeof(a))
    end
    if !already_owned
        mem = HostMemory(ctx, ptr, sizeof(a))
        resource_finalizer(a; blocking=false, ctx=nothing) do a
            # keep the object alive until its memory has been unregistered
            release_registration(RetiredRegistration(mem, true, a))
        end
    end

    a
end

function pin(ref::Base.RefValue{T}) where T
    ctx = context()
    ptr = Base.unsafe_convert(Ptr{T}, ref)

    sz = aligned_sizeof(T)
    __pin(ptr, sz)
    mem = HostMemory(ctx, ptr, sz)
    resource_finalizer(ref; blocking=false, ctx=nothing) do ref
        release_registration(RetiredRegistration(mem, true, ref))
    end

    ref
end

# derived arrays should always pin the parent memory range, because we may end up copying
# from or to that parent range (containing the derived range), and partially-pinned ranges
# are not supported:
#
# > Memory regions requested must be either entirely registered with CUDA, or in the case
# > of host pageable transfers, not registered at all. Memory regions spanning over
# > allocations that are both registered and not registered with CUDA are not supported and
# > will return CUDA_ERROR_INVALID_VALUE.
__pin(a::Union{SubArray, Base.ReinterpretArray, Base.ReshapedArray}) = __pin(parent(a))

# refcount the pinning per context, since we can only pin a memory range once
const __pinned_memory = Dict{Tuple{CuContext,Ptr{Cvoid}}, HostMemory}()
const __pin_count = Dict{Tuple{CuContext,Ptr{Cvoid}}, Int}()
function __pin(ptr::Ptr, sz::Int)
    ctx = context()
    key = (ctx, convert(Ptr{Nothing}, ptr))

    maybe_unregister_pending!()
    try_pin(key, ptr, sz) && return
    unregister_pending!()
    try_pin(key, ptr, sz) || throw(CuError(ERROR_HOST_MEMORY_ALREADY_REGISTERED))
    return
end
# returns false if the memory is still registered (see `try_register`)
function try_pin(key, ptr::Ptr, sz::Int)
    Base.@lock pin_lock begin
        pin_count = get(__pin_count, key, 0)
        if pin_count == 0
            # only record the pin once the memory has been registered
            mem = try_register(ptr, sz)
            mem === nothing && return false
            __pinned_memory[key] = mem
        elseif Base.JLOptions().debug_level >= 2
            @assert sz == sizeof(__pinned_memory[key])
        end
        __pin_count[key] = pin_count + 1
    end
    return true
end
function __unpin(ptr::Ptr, ctx::CuContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))

    mem = Base.@lock pin_lock begin
        @assert haskey(__pin_count, key) "Cannot unpin unmanaged pointer $ptr."
        pin_count = __pin_count[key] -= 1
        pin_count == 0 || return
        delete!(__pin_count, key)
        pop!(__pinned_memory, key)
    end
    try
        context!(ctx) do
            unregister(mem)
        end
    catch
        # the memory is still registered
        Base.@lock pin_lock begin
            __pinned_memory[key] = mem
            __pin_count[key] = get(__pin_count, key, 0) + 1
        end
        rethrow()
    end

    return
end
function __pinned(ptr::Ptr, ctx::CuContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))
    Base.@lock pin_lock begin
        haskey(__pin_count, key)
    end
end
