module CuArrays

using CUDAapi, CUDAdrv, CUDAnative

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

import LinearAlgebra

using Adapt

using Libdl

using Requires


## source code includes

# core array functionality
include("memory.jl")
include("array.jl")
include("gpuarrays.jl")
include("subarray.jl")
include("utils.jl")

# integrations and specialized functionality
include("permuteddimsarray.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")
include("nnlib.jl")

# vendor libraries
include("blas/CUBLAS.jl")
include("sparse/CUSPARSE.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("dnn/CUDNN.jl")
include("tensor/CUTENSOR.jl")

include("deprecated.jl")


## initialization

const __initialized__ = Ref(false)
functional() = __initialized__[]

export has_cudnn, has_cutensor
has_cudnn() = Libdl.dlopen_e(CUDNN.libcudnn[]) !== C_NULL
has_cutensor() = Libdl.dlopen_e(CUTENSOR.libcutensor[]) !== C_NULL

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional() || !CUDAnative.functional()
        verbose && @warn "CuArrays.jl did not initialize because CUDAdrv.jl or CUDAnative.jl failed to"
        return
    end

    try
        # discover libraries
        toolkit = find_toolkit()
        for name in ("cublas", "cusparse", "cusolver", "cufft", "curand", "cudnn", "cutensor")
            mod = getfield(CuArrays, Symbol(uppercase(name)))
            lib = Symbol("lib$name")
            handle = getfield(mod, lib)

            # on Windows, the library name is version dependent
            if Sys.iswindows()
                cuda = CUDAnative.version()
                suffix = cuda >= v"10.1" ? "$(cuda.major)" : "$(cuda.major)$(cuda.minor)"
                handle[] = "$(name)$(Sys.WORD_SIZE)_$(suffix)"
            end

            # check if we can't find the library
            if Libdl.dlopen_e(handle[]) == C_NULL
                path = find_cuda_library(name, toolkit)
                if path !== nothing
                    handle[] = path
                end
            end
        end

        # library dependencies
        CUBLAS.version()
        CUSPARSE.version()
        CUSOLVER.version()
        CUFFT.version()
        CURAND.version()
        # CUDNN and CUTENSOR are optional

        # library compatibility
        if has_cutensor()
            cutensor = CUTENSOR.version()
            if cutensor < v"1"
                silent || @warn("CuArrays.jl only supports CUTENSOR 1.0 or higher")
            end

            cuda = CUDAnative.version()
            cutensor_cuda = CUTENSOR.cuda_version()
            if cutensor_cuda.major != cuda.major || cutensor_cuda.minor != cuda.minor
                silent || @warn("You are using CUTENSOR $cutensor for CUDA $cutensor_cuda with CUDA toolkit $cuda; these might be incompatible.")
            end
        end
        if has_cudnn()
            cudnn = CUDNN.version()
            if cudnn < v"7.6"
                silent || @warn("CuArrays.jl only supports CUDNN v7.6 or higher")
            end

            cuda = CUDAnative.version()
            cudnn_cuda = CUDNN.cuda_version()
            if cudnn_cuda.major != cuda.major || cudnn_cuda.minor != cuda.minor
                silent || @warn("You are using CUDNN $cudnn for CUDA $cudnn_cuda with CUDA toolkit $cuda; these might be incompatible.")
            end
        end

        # package integrations
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")

        __init_memory__()

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "CuArrays.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CuArrays.jl failed to initialize and will be unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end # module
