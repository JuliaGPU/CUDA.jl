module CUTENSOR

import CUDAdrv: CUDAdrv, CuContext, CuStream_t, CuPtr, PtrOrCuPtr, CU_NULL
import CUDAapi

using Libdl
using CuArrays
Libdl.dlopen(CuArrays.CUBLAS.libcublas, RTLD_NOW | RTLD_DEEPBIND | RTLD_GLOBAL)
Libdl.dlopen(libcutensor, RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL)
using CuArrays: active_context

using LinearAlgebra
include("libcutensor_types.jl")
include("error.jl")
const _handles = Dict{CuContext,cutensorHandle_t}()
const _handle = Ref{cutensorHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cutensorCreate()
            atexit(()->CUDAdrv.isvalid(context) && cutensorDestroy(handle))
            handle
        end
    end
    return _handle[]
end
include("libcutensor.jl")
include("highlevel.jl")
include("wrappers.jl")

export CuTensor

version() = VersionNumber(cutensorGetProperty(CUDAapi.MAJOR_VERSION),
                          cutensorGetProperty(CUDAapi.MINOR_VERSION),
                          cutensorGetProperty(CUDAapi.PATCH_LEVEL))

end
