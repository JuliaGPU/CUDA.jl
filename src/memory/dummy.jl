module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CuArrays: @pool_timeit, actual_alloc, actual_free

using CUDAdrv

init() = return

deinit() = @assert isempty(allocated) "Cannot deinitialize memory pool with outstanding allocations"

const allocated = Set{Mem.Buffer}()

function alloc(sz)
    buf = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc(false)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc(true)" GC.gc(true)
        end

        @pool_timeit "$phase.1 alloc" begin
            buf = actual_alloc(sz)
        end
        buf === nothing || break
    end

    if buf !== nothing
        push!(allocated, buf)
        return buf
    end
end

function free(buf)
    delete!(allocated, buf)
    actual_free(buf)
    return
end

used_memory() = isempty(allocated) ? 0 : sum(sizeof, allocated)

cached_memory() = 0

dump() = return

end
