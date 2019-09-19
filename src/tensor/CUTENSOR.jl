module CUTENSOR

using ..CuArrays
using ..CuArrays: libcutensor, @libcutensor, active_context

using CUDAapi

using CUDAdrv
using CUDAdrv: CuStream_t

using CEnum
const cudaDataType_t = cudaDataType

include("libcutensor_common.jl")
include("error.jl")

function version()
    ver = cutensorGetVersion()
    major, ver = divrem(ver, 10000)
    minor, patch = divrem(ver, 100)

    VersionNumber(major, minor, patch)
end

include("libcutensor.jl")
include("highlevel.jl")
include("wrappers.jl")

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
