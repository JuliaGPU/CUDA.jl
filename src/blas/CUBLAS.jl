module CUBLAS

using CUDAapi

using CUDAdrv: CUDAdrv, CuContext, CuStream_t, CuPtr, PtrOrCuPtr, CU_NULL, devices

import CUDAnative

using ..CuArrays
using ..CuArrays: libcublas, active_context, unsafe_free!
using LinearAlgebra

using CEnum

include("libcublas_common.jl")
include("error.jl")

const _handles = Dict{CuContext,cublasHandle_t}()
const _xt_handles = Dict{CuContext,cublasXtHandle_t}()
const _handle = Ref{cublasHandle_t}(C_NULL)
const _xt_handle = Ref{cublasXtHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        CUDAnative.maybe_initialize("CUBLAS")
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cublasCreate_v2()

            # enable tensor math mode if our device supports it, and fast math is enabled
            dev = CUDAdrv.device(context)
            if Base.JLOptions().fast_math == 1 && CUDAdrv.capability(dev) >= v"7.0" && version() >= v"9"
                cublasSetMathMode(CUBLAS_TENSOR_OP_MATH, handle)
            end

            atexit(()->CUDAdrv.isvalid(context) && cublasDestroy_v2(handle))
            handle
        end
    end

    return _handle[]
end

function xt_handle()
    if _xt_handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _xt_handle[] = get!(_xt_handles, active_context[]) do
            context = active_context[]
            handle = cublasXtCreate()
            devs = convert.(Cint, CUDAdrv.devices())
            cublasXtDeviceSelect(handle, length(devs), devs)
            atexit(()->CUDAdrv.isvalid(context) && cublasXtDestroy(handle))
            handle
        end
    end
    return _xt_handle[]
end

version() = VersionNumber(cublasGetProperty(CUDAapi.MAJOR_VERSION),
                          cublasGetProperty(CUDAapi.MINOR_VERSION),
                          cublasGetProperty(CUDAapi.PATCH_LEVEL))

include("libcublas.jl")
include("wrappers.jl")
include("util.jl")
include("highlevel.jl")
include("linalg.jl")

end
