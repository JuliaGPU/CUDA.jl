module CUDNN

using CUDAapi
using CUDAapi: libraryPropertyType

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

using ..CuArrays
using ..CuArrays: active_context, @argout, @workspace
import ..CuArrays.unsafe_free!

import NNlib

const libcudnn = Ref("libcudnn")

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

end
