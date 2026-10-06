@public pin

# given object, find base allocation
# pin that, or increase refcount
# finalizer, drop refcount, free if 0

## memory pinning

const __pin_lock = ReentrantLock()

struct PinnedObject
    ref::WeakRef
    size::Int  # memory size in bytes
end

# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_objects = Dict{Tuple{CuContext,Ptr{Cvoid}}, PinnedObject}()

function pin(a::AbstractArray)
    ctx = context()
    ptr = pointer(a)

    Base.@lock __pin_lock begin
        # only pin an object once per context
        key = (ctx, convert(Ptr{Nothing}, ptr))
        previous = get(__pinned_objects, key, nothing)
        already_owned = previous !== nothing && previous.ref.value !== nothing
        if already_owned && haskey(__pin_count, key)
            sizeof(a) == previous.size && return nothing
            # the object was resized: replace its registration, but not its finalizer
            __unpin(ptr, ctx)
        end
        __pin(ptr, sizeof(a))
        __pinned_objects[key] = PinnedObject(WeakRef(a), sizeof(a))
        if !already_owned
            finalizer(a) do _
                __unpin(ptr, ctx)
            end
        end
    end

    a
end

function pin(ref::Base.RefValue{T}) where T
    ctx = context()
    ptr = Base.unsafe_convert(Ptr{T}, ref)

    __pin(ptr, aligned_sizeof(T))
    finalizer(ref) do _
        __unpin(ptr, ctx)
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

    Base.@lock __pin_lock begin
        pin_count = get(__pin_count, key, 0)
        if pin_count == 0
            # only record the pin once the memory has been registered
            __pinned_memory[key] = register(HostMemory, ptr, sz)
        elseif Base.JLOptions().debug_level >= 2
            # make sure we're pinning the exact same range
            mem = __pinned_memory[key]
            @assert sz == sizeof(mem) "Mismatch between pin request of $ptr: $sz vs. $(sizeof(mem))."
        end
        __pin_count[key] = pin_count + 1
    end

    return
end
function __unpin(ptr::Ptr, ctx::CuContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))

    Base.@lock __pin_lock begin
        @assert haskey(__pin_count, key) "Cannot unpin unmanaged pointer $ptr."
        pin_count = __pin_count[key] - 1

        if pin_count == 0
            # only forget about the pin once the memory has been unregistered
            mem = @inbounds __pinned_memory[key]
            context!(ctx) do
                unregister(mem)
            end
            delete!(__pinned_memory, key)
            delete!(__pin_count, key)
        else
            __pin_count[key] = pin_count
        end
    end

    return
end
function __pinned(ptr::Ptr, ctx::CuContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))
    Base.@lock __pin_lock begin
        haskey(__pin_count, key)
    end
end
