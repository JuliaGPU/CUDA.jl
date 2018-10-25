module CUDNN

using CUDAdrv

using ..CuArrays: CuArray, libcudnn, configured, active_context

include("libcudnn_types.jl")

const _handles = Dict{CuContext,cudnnHandle_t}()
const _handle = Ref{cudnnHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            handle = cudnnCreate()
            atexit(()->cudnnDestroy(handle))
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
