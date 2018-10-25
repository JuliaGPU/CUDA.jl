module CUBLAS

using CUDAdrv
import CUDAdrv: CuStream_t

using LinearAlgebra

using ..CuArrays: CuArray, CuVector, CuMatrix, CuVecOrMat,
                  libcublas, active_context

include("libcublas_types.jl")

const _handles = Dict{CuContext,cublasHandle_t}()
const _handle = Ref{cublasHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            handle = cublasCreate_v2()
            atexit(()->cublasDestroy_v2(handle))
            handle
        end
    end

    return _handle[]
end

include("error.jl")
include("util.jl")
include("libcublas.jl")
include("wrap.jl")
include("highlevel.jl")

end
