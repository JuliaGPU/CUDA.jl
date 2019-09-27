module CUDNN

using CUDAapi
using CUDAapi: libraryPropertyType

using CUDAdrv
using CUDAdrv: CuContext, CuPtr, PtrOrCuPtr, CU_NULL, CuStream_t

import CUDAnative

using CEnum

using ..CuArrays
using ..CuArrays: @libcudnn, active_context, CuVecOrMat, CuVector
import ..CuArrays.unsafe_free!

import NNlib

include("libcudnn_common.jl")
include("error.jl")

const _handles = Dict{CuContext,cudnnHandle_t}()
const _handle = Ref{cudnnHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        CUDAnative.maybe_initialize("CUDNN")
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cudnnCreate()
            atexit(()->CUDAdrv.isvalid(context) && cudnnDestroy(handle))
            handle
        end
    end

    return _handle[]
end

include("base.jl")
include("libcudnn.jl")

include("helpers.jl")
include("tensor.jl")
include("conv.jl")
include("pooling.jl")
include("activation.jl")
include("filter.jl")
include("softmax.jl")
include("batchnorm.jl")
include("dropout.jl")
include("rnn.jl")

# interfaces with other software
include("nnlib.jl")

include("compat.jl")

end
