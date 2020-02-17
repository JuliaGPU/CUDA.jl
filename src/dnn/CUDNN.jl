module CUDNN

using CUDAapi
using CUDAapi: libraryPropertyType

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

using ..CuArrays
using ..CuArrays: libcudnn, @argout, @workspace
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

const created_handles = IdDict{CuContext,cudnnHandle_t}()
const active_handles = Vector{Union{Nothing,cudnnHandle_t}}()

function handle()
    tid = Threads.threadid()
    if @inbounds active_handles[tid] === nothing
        ctx = context()
        active_handles[tid] = get!(created_handles, ctx) do
            handle = cudnnCreate()
            atexit(()->CUDAdrv.isvalid(ctx) && cudnnDestroy(handle))
            handle
        end
    end
    @inbounds active_handles[tid]
end

function __init__()
    resize!(active_handles, Threads.nthreads())
    fill!(active_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        # we don't eagerly initialize handles, but do so lazily when requested
        active_handles[tid] = nothing
    end
end

end
