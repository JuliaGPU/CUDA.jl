module cuSPARSE

using CUDACore
using GPUToolbox

using CUDACore: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using CUDACore: unsafe_free!, retry_reclaim, initialize_context, @allowscalar

using GPUArrays

using CEnum: @cenum

using LinearAlgebra
using LinearAlgebra: HermOrSym

using Adapt: Adapt, adapt

using SparseArrays

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end

const SparseChar = Char


@public functional

const _initialized = Ref{Bool}(false)
functional() = _initialized[]

# core library
include("libcusparse.jl")

include("error.jl")
include("array.jl")
include("util.jl")
include("types.jl")
include("linalg.jl")


# low-level wrappers
include("helpers.jl")
include("management.jl")
include("level2.jl")
include("level3.jl")
include("extra.jl")
include("preconditioners.jl")
include("reorderings.jl")
include("conversions.jl")
include("generic.jl")

# high-level integrations
include("interfaces.jl")

include("batched.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cusparseCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cusparseDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cusparseHandle_t}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound finalizer:
# when TLS state is cleared (e.g. on reclaim) and GC runs, the wrapper is
# collected and its finalizer returns the handle to the idle cache instead
# of the handle being pinned for the entire lifetime of the owning task.
mutable struct cusparseHandle
    const handle::cusparseHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cusparseHandle_t}, h::cusparseHandle) = h.handle

function handle_finalizer(h::cusparseHandle)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::cusparseHandle, stream::CuStream}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:CUSPARSE)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        wrapped = cusparseHandle(new_handle, cuda.context)
        finalizer(handle_finalizer, wrapped)

        cusparseSetStream(new_handle, cuda.stream)

        (; handle=wrapped, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cusparseSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end


function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcusparse
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cusparse"; optional=true)
        if path === nothing
            precompiling || @error "cuSPARSE is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcusparse = path
    else
        libcusparse = CUDA_Runtime_jll.libcusparse
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

# KernelAbstractions integration
import KernelAbstractions as KA
KA.get_backend(::CuSparseVector)    = CUDACore.CUDAKernels.CUDABackend()
KA.get_backend(::CuSparseMatrixCSC) = CUDACore.CUDAKernels.CUDABackend()
KA.get_backend(::CuSparseMatrixCSR) = CUDACore.CUDAKernels.CUDABackend()

include("precompile.jl")

# deprecated binding for backwards compatibility
Base.@deprecate_binding CUSPARSE cuSPARSE false

end
