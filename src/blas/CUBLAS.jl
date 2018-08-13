module BLAS

using CUDAdrv
using CUDAnative

using LinearAlgebra

using ..CuArrays: CuArray, CuVector, CuMatrix, CuVecOrMat, libcublas, configured

include("util.jl")
include("libcublas_types.jl")
include("error.jl")

# Typedef needed by libcublas
const cudaStream_t = Ptr{Nothing}

include("libcublas.jl")

const libcublas_handles = Dict{CuContext,cublasHandle_t}()
const libcublas_handle = Ref{cublasHandle_t}(C_NULL)

function __init__()
    configured || return

    # initialize the library when we switch devices
    callback = (dev::CuDevice, ctx::CuContext) -> begin
        libcublas_handle[] = get!(libcublas_handles, ctx) do
            @debug "Initializing CUBLAS for $dev"
            handle = Ref{cublasHandle_t}()
            cublasCreate_v2(handle)
            handle[]
        end
    end
    push!(CUDAnative.device!_listeners, callback)

    # deinitialize when exiting
    atexit() do
        libcublas_handle[] = C_NULL

        for (ctx, handle) in libcublas_handles
            if CUDAdrv.isvalid(ctx)
                cublasDestroy_v2(handle)
            end
        end
    end
end

include("wrap.jl")
include("highlevel.jl")

end
