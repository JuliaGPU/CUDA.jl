module CUTENSOR

using ..CuArrays
using ..CuArrays: @argout, @workspace

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

const cudaDataType_t = cudaDataType

const libcutensor = Ref("libcutensor")

# core library
include("libcutensor_common.jl")
include("error.jl")
include("libcutensor.jl")

# low-level wrappers
include("tensor.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

const created_handles = IdDict{CuContext,Ref{cutensorHandle_t}}()
const active_handles = Vector{Union{Nothing,Ref{cutensorHandle_t}}}()

function handle()
    tid = Threads.threadid()
    if @inbounds active_handles[tid] === nothing
        ctx = context()
        active_handles[tid] = get!(created_handles, ctx) do
            handle = Ref{cutensorHandle_t}()
            cutensorInit(handle)
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
