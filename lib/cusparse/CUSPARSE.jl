module CUSPARSE

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusparse, unsafe_free!, @retry_reclaim, @context!

using CEnum

using Memoize

using LinearAlgebra
using LinearAlgebra: HermOrSym

using Adapt

using DataStructures

using SparseArrays

const SparseChar = Char


# core library
include("libcusparse_common.jl")
include("error.jl")
include("libcusparse.jl")
include("libcusparse_deprecated.jl")

include("array.jl")
include("util.jl")
include("types.jl")

# low-level wrappers
include("helpers.jl")
include("management.jl")
include("level1.jl")
include("level2.jl")
include("level3.jl")
include("preconditioners.jl")
include("conversions.jl")
include("generic.jl")

# high-level integrations
include("interfaces.jl")

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,cusparseHandle_t}()

function handle()
    state = CUDA.active_state()
    handle, stream = get!(task_local_storage(), (:CUSPARSE, state.context)) do
        new_handle = pop!(idle_handles, state.context) do
            cusparseCreate()
        end

        finalizer(current_task()) do task
            push!(idle_handles, state.context, new_handle) do
                @context! skip_destroyed=true state.context cusparseDestroy(new_handle)
            end
        end
        # TODO: cusparseDestroy to preserve memory, or at exit?

        cusparseSetStream(new_handle, state.stream)

        new_handle, state.stream
    end::Tuple{cusparseHandle_t,CuStream}

    if stream != state.stream
        cusparseSetStream(handle, state.stream)
        task_local_storage((:CUSPARSE, state.context), (handle, state.stream))
    end

    return handle
end

end
