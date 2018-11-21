module CUBLAS

import CUDAdrv: CUDAdrv, CuContext, CuStream_t
import CUDAapi

using ..CuArrays
using ..CuArrays: libcublas, active_context

using LinearAlgebra

include("libcublas_types.jl")
include("error.jl")

const _handles = Dict{CuContext,cublasHandle_t}()
const _handle = Ref{cublasHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cublasCreate_v2()
            atexit(()->CUDAdrv.isvalid(context) && cublasDestroy_v2(handle))
            handle
        end
    end

    return _handle[]
end

include("libcublas.jl")
include("util.jl")
include("wrappers.jl")
include("highlevel.jl")

version() = VersionNumber(cublasGetProperty(CUDAapi.MAJOR_VERSION),
                          cublasGetProperty(CUDAapi.MINOR_VERSION),
                          cublasGetProperty(CUDAapi.PATCH_LEVEL))

end
