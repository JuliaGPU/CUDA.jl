module cuSPARSE

using CUDA
using CUDA.APIUtils
using GPUToolbox

using CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using CUDA: unsafe_free!, retry_reclaim, initialize_context, @allowscalar

using GPUArrays

using CEnum: @cenum

using LinearAlgebra
using LinearAlgebra: HermOrSym

using Adapt: Adapt, adapt

using SparseArrays

if CUDA.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end

const SparseChar = Char


export has_cusparse

const _initialized = Ref{Bool}(false)
has_cusparse() = _initialized[]

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

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cusparseHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUSPARSE) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle)
        end

        cusparseSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream)
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

    CUDA.functional() || return

    # find the library
    global libcusparse
    if CUDA.local_toolkit
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

    _initialized[] = true
end

# KernelAbstractions integration
import KernelAbstractions as KA
KA.get_backend(::CuSparseVector)    = CUDA.CUDAKernels.CUDABackend()
KA.get_backend(::CuSparseMatrixCSC) = CUDA.CUDAKernels.CUDABackend()
KA.get_backend(::CuSparseMatrixCSR) = CUDA.CUDAKernels.CUDABackend()

# deprecated binding for backwards compatibility
Base.@deprecate_binding CUSPARSE cuSPARSE false

end
