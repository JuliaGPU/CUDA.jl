# host-side functionality for receiving method calls from the GPU

const HOSTCALL_POOL_SIZE = UInt32(1024*16)   # ~64MB
# ring buffer helpers assume pow2
@assert ispow2(HOSTCALL_POOL_SIZE)
# we should be able to request slots for a full warp, or we would deadlock
@assert HOSTCALL_POOL_SIZE >= 32
# head and tail pointers can exceed HOSTCALL_POOL_SIZE, so overflow behaviour should match
@assert (typemax(UInt32)+1)%HOSTCALL_POOL_SIZE == 0

struct HostcallPool
    context::CuContext

    # mapped host storage for ring buffer pointers
    #
    # we can't perform operations that are atomic wrt. both the CPU and GPU, only wrt. to
    # a single device, but that's okay as the tail pointer is only moved by the CPU, while
    # the head pointer is only moved by the GPU. stale reads from either device will only
    # result in under-estimated capacities.
    pointer_buf::Mem.HostBuffer
    pointers::Vector{UInt32}   # [head, tail], 0-indexed for simplified modulo arithmetic

    # mapped host storage for actual hostcall objects
    call_buf::Mem.HostBuffer
    calls::Vector{Hostcall}
end

# small helpers for pow2 ring buffer management.
# - the head is where the producer inserts, the tail is where the consumer reads
# - tail == head indicates an empty buffer
# - head and tail pointers can be 0 or 1 indexed, and do not need to fall within size bounds
ring_count(head, tail, size) = (head - tail) & (size-1)
ring_space(head, tail, size) = ring_count(tail, head+1, size)
# NOTE: one item is left unused, as a full buffer means head==tail which also means empty

# create and return the hostcall pool for each context
const hostcall_pools = Dict{CuContext, HostcallPool}()
hostcall_pool(ctx::CuContext) = get!(hostcall_pools, ctx) do
    @context! ctx begin
        # NOTE: we allocate the host memory manually, instead of just registering an array,
        #       to avoid accidentally re-registering a memory range.
        pointer_buf = Mem.alloc(Mem.Host, 2*sizeof(UInt32), Mem.HOSTALLOC_DEVICEMAP)
        pointer_ptr = convert(Ptr{UInt32}, pointer_buf)
        pointers = unsafe_wrap(Array, pointer_ptr, 2)
        fill!(pointers, 0)

        call_buf = Mem.alloc(Mem.Host, HOSTCALL_POOL_SIZE*sizeof(Hostcall), Mem.HOSTALLOC_DEVICEMAP)
        call_ptr = convert(Ptr{Hostcall}, call_buf)
        calls = unsafe_wrap(Array, call_ptr, HOSTCALL_POOL_SIZE)

        pool = HostcallPool(ctx, pointer_buf, pointers, call_buf, calls)
        marker = Threads.Atomic{Int}(0)

        watcher = @async begin
            while isvalid(ctx)
                Base.invokelatest(check_hostcalls, pool)
                marker[] = 1
                sleep(0.1)
            end
        end
        VERSION >= v"1.7-" && errormonitor(watcher)

        hostcall_markers[ctx] = marker
        return pool
    end
end

# wait for all hostcalls to complete.
# XXX: add to `synchronize()`?
const hostcall_markers = Dict{CuContext, Threads.Atomic{Int}}()
function hostcall_synchronize(ctx::CuContext=context())
    haskey(hostcall_pools, ctx) || return
    marker = hostcall_markers[ctx]
    marker[] = 0
    while marker[] == 0
        sleep(0.1)
    end
    return
end

# check whether a pool has any outstanding hostcalls, and execute them
function check_hostcalls(pool::HostcallPool)
    head0, tail0 = pool.pointers
    while ring_count(head0, tail0, HOSTCALL_POOL_SIZE) >= 1
        slot = tail0 & (HOSTCALL_POOL_SIZE - 0x1) + 0x1
        hostcall = pool.calls[slot]
        hostcall_ptr = pointer(pool.calls, slot)

        if hostcall.state == HOSTCALL_SUBMITTED
            # Setfield.jl chokes on the 4k tuple, so we manually create pointers to fields.
            state_ptr = reinterpret(Ptr{HostcallState}, hostcall_ptr) + fieldoffset(Hostcall, 1)
            buffer_ptr = hostcall_ptr + fieldoffset(Hostcall, fieldcount(Hostcall))

            try
                sig, rettyp = hostcall_targets[hostcall.target]
                # function barrier for specialization
                state = process_hostcall(sig, rettyp, buffer_ptr)
                unsafe_store!(state_ptr, state)
            catch ex
                Base.display_error(ex, catch_backtrace())
                unsafe_store!(state_ptr, HOSTCALL_READY)
            end
        end

        tail0 += 0x1
        pool.pointers[2] = tail0
    end
end

@inline @generated function read_hostcall_arguments(ptr, sig)
    args = []
    last_offset = 0
    for typ in sig.parameters
        sz = sizeof(typ)
        arg = if sz > 0
            align = Base.datatype_alignment(typ)
            offset = Base.cld(last_offset, align) * align
            last_offset = offset + sz
            if last_offset > HOSTCALL_BUFFER_SIZE
                return :(error("hostcall arguments exceed maximum buffer size"))
            end
            :(unsafe_load(reinterpret(Ptr{$typ}, ptr+$offset)))
        else
            :($(typ.instance))
        end
        push!(args, arg)
    end

    quote
        ($(args...))
    end
end

@noinline function process_hostcall(sig::Type{T}, rettyp::Type{U}, buffer_ptr) where {T,U}
    f, args... = read_hostcall_arguments(buffer_ptr, sig)
    rv = Base.invokelatest(f, args...)::rettyp

    if rettyp === Nothing
        HOSTCALL_READY
    else
        # store the return type
        if sizeof(rettyp) > HOSTCALL_BUFFER_SIZE
            error("hostcall return value exceeds maximum buffer size")
        end
        unsafe_store!(reinterpret(Ptr{rettyp}, buffer_ptr), rv)
        HOSTCALL_RETURNED
    end
end
