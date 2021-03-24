module CUSPARSE

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusparse, unsafe_free!, @retry_reclaim

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
const handle_cache_lock = ReentrantLock()
const idle_handles = DefaultDict{CuContext,Vector{cusparseHandle_t}}(()->cusparseHandle_t[])

function handle()
    ctx = context()
    active_stream = stream()
    handle, chosen_stream = get!(task_local_storage(), (:CUSPARSE, ctx)) do
        new_handle = @lock handle_cache_lock begin
            if isempty(idle_handles[ctx])
                cusparseCreate()
            else
                pop!(idle_handles[ctx])
            end
        end

        finalizer(current_task()) do task
            @spinlock handle_cache_lock begin
                push!(idle_handles[ctx], new_handle)
            end
        end
        # TODO: cusparseDestroy to preserve memory, or at exit?

        cusparseSetStream(new_handle, active_stream)

        new_handle, active_stream
    end::Tuple{cusparseHandle_t,CuStream}

    if chosen_stream != active_stream
        cusparseSetStream(handle, active_stream)
        task_local_storage((:CUSPARSE, ctx), (handle, active_stream))
    end

    return handle
end

end
