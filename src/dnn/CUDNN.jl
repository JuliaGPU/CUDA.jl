module CUDNN

import CUDAapi

import CUDAdrv: CUDAdrv, CuContext, CuPtr, CU_NULL

using ..CuArrays
using ..CuArrays: @libcudnn, active_context, unsafe_free!
using ..CuArrays: CuVecOrMat, CuVector

using NNlib
import NNlib: conv!, ∇conv_filter!, ∇conv_data!, stride, dilation, flipkernel,
  maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, spatial_dims, padding, kernel_size,
  softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax

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
include("compat.jl")

version() = VersionNumber(cudnnGetProperty(CUDAapi.MAJOR_VERSION),
                          cudnnGetProperty(CUDAapi.MINOR_VERSION),
                          cudnnGetProperty(CUDAapi.PATCH_LEVEL))

end
