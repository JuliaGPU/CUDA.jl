module CUBLAS

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using ..CuArrays
using ..CuArrays: unsafe_free!
using LinearAlgebra

using CEnum

const libcublas = Ref("libcublas")

# core library
include("libcublas_common.jl")
include("error.jl")
include("libcublas.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("linalg.jl")

const created_handles = IdDict{CuContext,cublasHandle_t}()
const created_xt_handles = IdDict{CuContext,cublasXtHandle_t}()
const active_handles = Vector{Union{Nothing,cublasHandle_t}}()
const active_xt_handles = Vector{Union{Nothing,cublasXtHandle_t}}()

function handle()
    tid = Threads.threadid()
    if @inbounds active_handles[tid] === nothing
        ctx = context()
        active_handles[tid] = get!(created_handles, ctx) do
            handle = cublasCreate_v2()
            atexit(()->CUDAdrv.isvalid(ctx) && cublasDestroy_v2(handle))

            # enable tensor math mode if our device supports it, and fast math is enabled
            dev = CUDAdrv.device()
            if Base.JLOptions().fast_math == 1 && CUDAdrv.capability(dev) >= v"7.0" && version() >= v"9"
                cublasSetMathMode(CUBLAS_TENSOR_OP_MATH, handle)
            end

            handle
        end
    end
    @inbounds active_handles[tid]
end

function xt_handle()
    tid = Threads.threadid()
    if @inbounds active_xt_handles[tid] === nothing
        ctx = context()
        active_xt_handles[tid] = get!(created_xt_handles, ctx) do
            handle = cublasXtCreate()
            atexit(()->CUDAdrv.isvalid(ctx) && cublasXtDestroy(handle))

            # select the devices
            # TODO: this is weird, since we typically use a single device per thread/context
            devs = convert.(Cint, CUDAdrv.devices())
            cublasXtDeviceSelect(handle, length(devs), devs)

            handle
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
