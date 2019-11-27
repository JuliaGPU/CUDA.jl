module CUDNN

using CUDAapi
using CUDAapi: libraryPropertyType

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

using ..CuArrays
using ..CuArrays: active_context, @workspace
import ..CuArrays.unsafe_free!

import NNlib

const libcudnn = if Sys.iswindows()
    # no ccall by soname, we need the filename
    # NOTE: we discover the full path here, while only the wordsize and toolkit versions
    #       would have been enough to construct "cudnn64_10.dll"
    toolkit = find_toolkit()
    path = find_cuda_library("cudnn", toolkit)
    if path === nothing
        nothing
    else
        basename(path)
    end
else
    # ccall by soname; CuArrays.__init__ will have populated Libdl.DL_LOAD_PATH
    "libcudnn"
end

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
