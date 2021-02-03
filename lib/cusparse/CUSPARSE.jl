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

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,cusparseHandle_t}}()

# cache for created, but unused handles
const old_handles = DefaultDict{CuContext,Vector{cusparseHandle_t}}(()->cusparseHandle_t[])

function handle()
    CUDA.detect_state_changes()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUSPARSE, ctx)) do
            handle = if isempty(old_handles[ctx])
                cusparseCreate()
            else
                pop!(old_handles[ctx])
            end

            finalizer(current_task()) do task
                push!(old_handles[ctx], handle)
            end
            # TODO: cusparseDestroy to preserve memory, or at exit?

            cusparseSetStream(handle, stream())

            handle
        end
    end
    something(@inbounds thread_handles[tid])
end

@inline function set_stream(stream::CuStream)
    ctx = context()
    tls = task_local_storage()
    handle = get(tls, (:CUSPARSE, ctx), nothing)
    if handle !== nothing
        cusparseSetStream(handle, stream)
    end
    return
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end
end

end
