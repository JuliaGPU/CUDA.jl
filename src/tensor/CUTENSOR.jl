module CUTENSOR

using ..CuArrays
using ..CuArrays: active_context

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CEnum
const cudaDataType_t = cudaDataType

const libcutensor = Ref("libcutensor")

# core library
include("libcutensor_common.jl")
include("error.jl")
include("libcutensor.jl")

# low-level wrappers
include("tensor.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

const _handles = Dict{CuContext,Ref{cutensorHandle_t}}()
const _handle = Ref{Union{Ref{cutensorHandle_t},Nothing}}(nothing)

function handle()
    if _handle[] == nothing
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = Ref{cutensorHandle_t}()
            cutensorInit(handle)
            handle
        end
    end
    return _handle[]
end

end
