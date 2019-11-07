module CUTENSOR

using ..CuArrays
using ..CuArrays: active_context

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CEnum
const cudaDataType_t = cudaDataType

const libcutensor = if Sys.iswindows()
    # no ccall by soname, we need the filename
    # NOTE: we discover the full path here, while only the wordsize and toolkit versions
    #       would have been enough to construct "cutensor64_10.dll"
    toolkit = find_toolkit()
    path = find_cuda_library("cutensor", toolkit)
    if path === nothing
        nothing
    else
        basename(path)
    end
else
    # ccall by soname; CuArrays.__init__ will have populated Libdl.DL_LOAD_PATH
    "libcutensor"
end

# core library
include("libcutensor_common.jl")
include("error.jl")
include("libcutensor.jl")

# low-level wrappers
include("tensor.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

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

end
