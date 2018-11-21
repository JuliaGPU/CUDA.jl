module CUDNN

import CUDAdrv: CUDAdrv, CuContext
import CUDAapi

using ..CuArrays
using ..CuArrays: libcudnn, active_context, configured

include("libcudnn_types.jl")
include("error.jl")

const _handles = Dict{CuContext,cudnnHandle_t}()
const _handle = Ref{cudnnHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cudnnCreate()
            atexit(()->CUDAdrv.isvalid(context) && cudnnDestroy(handle))
            handle
        end
    end

    return _handle[]
end

include("libcudnn.jl")
include("helpers.jl")
include("nnlib.jl")

version() = VersionNumber(cudnnGetProperty(CUDAapi.MAJOR_VERSION),
                          cudnnGetProperty(CUDAapi.MINOR_VERSION),
                          cudnnGetProperty(CUDAapi.PATCH_LEVEL))

end
