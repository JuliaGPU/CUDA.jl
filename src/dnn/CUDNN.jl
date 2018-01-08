module CUDNN

using ..CuArrays: CuArray, libcudnn, configured

include("libcudnn_types.jl")
include("error.jl")
include("helpers.jl")
include("libcudnn.jl")
include("nnlib.jl")

const libcudnn_handle = Ref{cudnnHandle_t}()
function __init__()
    configured || return

    cudnnCreate(libcudnn_handle)
    atexit(()->cudnnDestroy(libcudnn_handle[]))

    global CUDNN_VERSION = convert(Int, ccall((:cudnnGetVersion,libcudnn),Csize_t,()))
end

end
