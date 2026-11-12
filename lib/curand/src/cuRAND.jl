module cuRAND

using CUDACore
using GPUToolbox
using CUDACore: CUstream, libraryPropertyType, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
using CUDACore: retry_reclaim, initialize_context, resource_finalizer

using CEnum: @cenum

using Random

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end


@public functional, version
@public rand, randn, seed!
@public rand_logn, rand_logn!, rand_poisson, rand_poisson!
@public LibraryRNG, NativeRNG, library_rng, native_rng, gpuarrays_rng

const _initialized = Ref{Bool}(false)
"""
    cuRAND.functional() -> Bool

Whether cuRAND was loaded and initialized successfully.
"""
functional() = _initialized[]

# core library
include("libcurand.jl")

# low-level wrappers
include("error.jl")
include("wrappers.jl")

# high-level integrations
include("random.jl")

# native kernel-based RNG
include("native.jl")

# CUDACore.rand/randn integration
include("cuda_integration.jl")

include("deprecated.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        LibraryRNG()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx) do
        # queue the generator's destruction, so that it is destroyed by the reclaim that
        # purges this cache, instead of only after it has been collected
        finalize(handle)
    end
end
const idle_library_rngs = HandleCache{CuContext,LibraryRNG}(handle_ctor, handle_dtor)

# wrapper owning a LibraryRNG borrowed from `idle_library_rngs`. Held in
# task-local storage so that, on reclaim or task GC, the wrapper becomes
# unreachable and its finalizer returns the RNG to the idle cache. From
# there, `purge!` on the cache would free the underlying generator.
mutable struct BorrowedLibraryRNG
    const rng::LibraryRNG
    const ctx::CuContext
end

function library_rng_finalizer(b::BorrowedLibraryRNG)
    push!(idle_library_rngs, b.ctx, b.rng)
end

const library_state_cache = CUDACore.TaskLocalCache{CuContext,
    CUDACore.TaskHandleState{BorrowedLibraryRNG, Tuple{}}}(:CURAND)

"""
    cuRAND.library_rng() -> LibraryRNG

The cuRAND generator of the current task, for its active context. `cuRAND.rand`, `randn`,
`rand_logn`, `rand_poisson` and the in-place `rand!`, `randn!`, `rand_logn!` and
`rand_poisson!` use it for the element types cuRAND supports. It is seeded randomly when a
task first uses it, and by [`cuRAND.seed!`](@ref). The generator is borrowed: seed it or
generate numbers with it, but don't destroy it or change its settings.
"""
function library_rng()
    ctx = CUDACore.active_state().context
    borrowed = CUDACore.task_handle(library_rng_finalizer, library_state_cache,
                                    idle_library_rngs, ctx) do rng
        Random.seed!(rng)
        BorrowedLibraryRNG(rng, ctx)
    end
    return borrowed.rng
end


function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcurand
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "curand"; optional=true)
        if path === nothing
            precompiling || @error "cuRAND is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcurand = path
    else
        libcurand = CUDA_Runtime_jll.libcurand
    end

    CUDACore.register_reclaimable!(idle_library_rngs)
    CUDACore.register_reclaimable!(library_state_cache)
    CUDACore.register_reclaimable!(native_state_cache)
    CUDACore.register_reclaimable!(gpuarrays_state_cache)

    _initialized[] = true
end

include("precompile.jl")

# deprecated binding for backwards compatibility
Base.@deprecate_binding CURAND cuRAND false

end
