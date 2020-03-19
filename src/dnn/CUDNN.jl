module CUDNN

using CUDAapi
using CUDAapi: libraryPropertyType

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

using ..CuArrays
using ..CuArrays: libcudnn, @argout, @workspace, @retry_reclaim
import ..CuArrays.unsafe_free!

import NNlib

# core library
include("libcudnn_common.jl")
include("error.jl")
include("libcudnn.jl")

# low-level wrappers
include("util.jl")
include("base.jl")
include("tensor.jl")
include("conv.jl")
include("pooling.jl")
include("activation.jl")
include("filter.jl")
include("softmax.jl")
include("batchnorm.jl")
include("dropout.jl")
include("rnn.jl")

# high-level integrations
include("nnlib.jl")

include("compat.jl")

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,cudnnHandle_t}}()

function handle()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUDNN, ctx)) do
            handle = cudnnCreate()
            atexit() do
                CUDAdrv.isvalid(ctx) || return
                context!(ctx) do
                    cudnnDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_handles[tid]
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        thread_handles[tid] = nothing
    end

    CUDAnative.attaskswitch() do tid, task
        thread_handles[tid] = nothing
    end
end

end
