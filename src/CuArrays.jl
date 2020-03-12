module CuArrays

using CUDAapi, CUDAdrv, CUDAnative

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, CuIterator, cu
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

import LinearAlgebra

using Adapt

using Libdl

using Requires


## deferred initialization

# CUDA packages require complex initialization (discover CUDA, download artifacts, etc)
# that can't happen at module load time, so defer that to run time upon actual use.

const configured = Ref{Union{Nothing,Bool}}(nothing)

"""
    functional(show_reason=false)

Check if the package has been configured successfully and is ready to use.

This call is intended for packages that support conditionally using an available GPU. If you
fail to check whether CUDA is functional, actual use of functionality might warn and error.
"""
function functional(show_reason::Bool=false)
    if configured[] === nothing
        _functional(show_reason)
    end
    configured[]::Bool
end

const configure_lock = ReentrantLock()
@noinline function _functional(show_reason::Bool=false)
    lock(configure_lock) do
        if configured[] === nothing
            if __configure__(show_reason)
                configured[] = true
                try
                    __runtime_init__()
                catch
                    configured[] = false
                    rethrow()
                end
            else
                configured[] = false
            end
        end
    end
end

# macro to guard code that only can run after the package has successfully initialized
macro after_init(ex)
    quote
        @assert functional(true) "CuArrays.jl did not successfully initialize, and is not usable."
        $(esc(ex))
    end
end


## source code includes

include("bindeps.jl")

# core array functionality
include("memory.jl")
include("array.jl")
include("gpuarrays.jl")
include("subarray.jl")
include("utils.jl")

# vendor libraries
include("blas/CUBLAS.jl")
include("sparse/CUSPARSE.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("dnn/CUDNN.jl")
include("tensor/CUTENSOR.jl")

# integrations and specialized functionality
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")
include("nnlib.jl")
include("iterator.jl")

include("deprecated.jl")

## initialization

function __init__()
    # package integrations
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")

    __init_memory__()
end

function __configure__(show_reason::Bool)
    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional(show_reason) || !CUDAnative.functional(show_reason)
        show_reason && @warn "CuArrays.jl did not initialize because CUDAdrv.jl or CUDAnative.jl failed to"
        return
    end

    return __configure_dependencies__(show_reason)
end

function __runtime_init__()
    cuda = version()

    if has_cutensor()
        cutensor = CUTENSOR.version()
        if cutensor < v"1"
             @warn("CuArrays.jl only supports CUTENSOR 1.0 or higher")
        end

        cutensor_cuda = CUTENSOR.cuda_version()
        if cutensor_cuda.major != cuda.major || cutensor_cuda.minor != cuda.minor
            @warn("You are using CUTENSOR $cutensor for CUDA $cutensor_cuda with CUDA toolkit $cuda; these might be incompatible.")
        end
    end

    if has_cudnn()
        cudnn = CUDNN.version()
        if cudnn < v"7.6"
            @warn("CuArrays.jl only supports CUDNN v7.6 or higher")
        end

        cudnn_cuda = CUDNN.cuda_version()
        if cudnn_cuda.major != cuda.major || cudnn_cuda.minor != cuda.minor
            @warn("You are using CUDNN $cudnn for CUDA $cudnn_cuda with CUDA toolkit $cuda; these might be incompatible.")
        end
    end

    return
end

end # module
