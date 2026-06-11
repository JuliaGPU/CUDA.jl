module cuRAND

using CUDACore
using GPUToolbox
using CUDACore: CUstream, libraryPropertyType, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
using CUDACore: retry_reclaim, initialize_context

using CEnum: @cenum

using Random

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end


@public functional
@public rand, randn, seed!
@public rand_logn, rand_logn!, rand_poisson, rand_poisson!
@public LibraryRNG, NativeRNG, library_rng, native_rng, gpuarrays_rng

const _initialized = Ref{Bool}(false)
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


## handles

function handle_ctor(ctx)
    context!(ctx) do
        LibraryRNG()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        # no need to do anything, as the RNG is collected by its finalizer
        # TODO: early free?
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

const library_state_cache = CUDACore.TaskLocalCache{CuContext, BorrowedLibraryRNG}(:CURAND)

function library_rng()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(library_state_cache)

    @noinline function new_state(cuda)
        new_rng = pop!(idle_library_rngs, cuda.context)
        wrapped = BorrowedLibraryRNG(new_rng, cuda.context)
        finalizer(library_rng_finalizer, wrapped)
        Random.seed!(new_rng)
        wrapped
    end
    borrowed = get!(states, cuda.context) do
        new_state(cuda)
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
