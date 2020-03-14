module CUBLAS

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using ..CuArrays
using ..CuArrays: libcublas, unsafe_free!, @retry_reclaim
using LinearAlgebra

using CEnum

# core library
include("libcublas_common.jl")
include("error.jl")
include("libcublas.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("linalg.jl")

const handles_lock = ReentrantLock()
const created_handles = Dict{Tuple{UInt,Int},cublasHandle_t}()
const created_xt_handles = Dict{Tuple{UInt,Int},cublasXtHandle_t}()
const active_handles = Vector{Union{Nothing,cublasHandle_t}}()
const active_xt_handles = Vector{Union{Nothing,cublasXtHandle_t}}()

function handle()
    tid = Threads.threadid()
    if @inbounds active_handles[tid] === nothing
        ctx = context()
        key = (objectid(ctx), tid)
        lock(handles_lock) do
            active_handles[tid] = get!(created_handles, key) do
                handle = cublasCreate_v2()
                atexit(()->CUDAdrv.isvalid(ctx) && cublasDestroy_v2(handle))

                # enable tensor math mode if our device supports it, and fast math is enabled
                dev = CUDAdrv.device()
                if Base.JLOptions().fast_math == 1 && CUDAdrv.capability(dev) >= v"7.0" && version() >= v"9"
                    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)
                end

                handle
            end
        end
    end
    @inbounds active_handles[tid]
end

function xt_handle()
    tid = Threads.threadid()
    if @inbounds active_xt_handles[tid] === nothing
        ctx = context()
        key = (objectid(ctx), tid)
        lock(handles_lock) do
            active_xt_handles[tid] = get!(created_xt_handles, key) do
                handle = cublasXtCreate()
                atexit(()->CUDAdrv.isvalid(ctx) && cublasXtDestroy(handle))

                # select the devices
                # TODO: this is weird, since we typically use a single device per thread/context
                devs = convert.(Cint, CUDAdrv.devices())
                cublasXtDeviceSelect(handle, length(devs), devs)

                handle
            end
        end
    end
    @inbounds active_xt_handles[tid]
end

function __init__()
    resize!(active_handles, Threads.nthreads())
    fill!(active_handles, nothing)

    resize!(active_xt_handles, Threads.nthreads())
    fill!(active_xt_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        # we don't eagerly initialize handles, but do so lazily when requested
        active_handles[tid] = nothing
        active_xt_handles[tid] = nothing
    end
end

end
