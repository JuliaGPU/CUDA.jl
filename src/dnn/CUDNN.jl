module CUDNN

import CUDAdrv: CUDAdrv, CuContext

using ..CuArrays
using ..CuArrays: libcudnn, active_context, configured

include("libcudnn_types.jl")

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

include("error.jl")
include("helpers.jl")
include("libcudnn.jl")
include("nnlib.jl")

function __init__()
    configured || return

    global CUDNN_VERSION = convert(Int, ccall((:cudnnGetVersion,libcudnn), Csize_t, ()))
end

end
