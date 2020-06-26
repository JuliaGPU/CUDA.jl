module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock, isvalid

using Base: @lock

const allocated_lock = NonReentrantLock()
const allocated = Dict{CuPtr{Nothing},Tuple{CuContext,Int}}()

init() = return

function alloc(sz, ctx=context())
    ptr = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 alloc" begin
            @assert isvalid(ctx)
            ptr = context!(ctx) do
                CUDA.actual_alloc(sz)
            end
        end
        ptr === nothing || break
    end

    if ptr !== nothing
        @safe_lock allocated_lock begin
            allocated[ptr] = (ctx,sz)
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    ctx, sz = @safe_lock_spin allocated_lock begin
        ctx, sz = allocated[ptr]
        delete!(allocated, ptr)
        ctx, sz
    end

    isvalid(ctx) || return
    context!(ctx) do
        CUDA.actual_free(ptr)
    end
    return
end

reclaim(target_bytes::Int=typemax(Int)) = return 0

used_memory() = @safe_lock allocated_lock mapreduce(sizeof, +, values(allocated); init=0)

cached_memory() = 0

end
