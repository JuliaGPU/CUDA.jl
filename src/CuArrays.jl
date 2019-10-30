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

include("memory.jl")
include("array.jl")
include("subarray.jl")
include("permuteddimsarray.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("matmul.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")

include("gpuarray_interface.jl")

# many libraries need to be initialized per-device (per-context, really, but we assume users
# of CuArrays and/or CUDAnative only use a single context), so keep track of the active one.
const active_context = Ref{CuContext}()

include("blas/CUBLAS.jl")
include("sparse/CUSPARSE.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("dnn/CUDNN.jl")
include("tensor/CUTENSOR.jl")

include("nnlib.jl")

include("deprecated.jl")


## initialization

function __init__()
    if ccall(:jl_generating_output, Cint, ()) == 1
        # don't initialize when we, or any package that depends on us, is precompiling.
        # this makes it possible to precompile on systems without CUDA,
        # at the expense of using the packages in global scope.
        return
    end

    # discovery
    toolkit = find_toolkit()
    ## required libraries that are part of the CUDA toolkit
    for name in ("cublas", "cusparse", "cusolver", "cufft", "curand")
        lib = Symbol("lib$name")
        path = find_cuda_library(name, toolkit)
        if path === nothing
            error("Could not find library '$name' (it should be part of the CUDA toolkit)")
        end
        dir = dirname(path)
        if !(dir in Libdl.DL_LOAD_PATH)
            push!(Libdl.DL_LOAD_PATH, dir)
        end
    end
    ## optional libraries
    for name in ("cudnn", "cutensor")
        lib = Symbol("lib$name")
        path = find_cuda_library(name, toolkit)
        if path !== nothing
            dir = dirname(path)
            if !(dir in Libdl.DL_LOAD_PATH)
                push!(Libdl.DL_LOAD_PATH, dir)
            end
        end

        # function to check for availability
        fn = Symbol("has_$name")
        @eval (export $fn; $fn() = $(path !== nothing))
    end

    # compiler barrier to avoid *seeing* `ccall`s to unavailable libraries
    Base.invokelatest(__hidden_init__)
end

function __hidden_init__()
    # compatibility
    if has_cutensor()
        ver = Base.invokelatest(CUTENSOR.version)
        if ver.major != 0 || ver.minor != 2
            error("CuArrays.jl only supports CUTENSOR 0.2")
        end
    end

    # package integrations
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")

    # update the active context when we switch devices
    callback = (::CuDevice, ctx::CuContext) -> begin
        active_context[] = ctx

        # wipe the active handles
        CUBLAS._handle[] = C_NULL
        CUBLAS._xt_handle[] = C_NULL
        CUSOLVER._dense_handle[] = C_NULL
        CUSOLVER._sparse_handle[] = C_NULL
        CUSPARSE._handle[] = C_NULL
        CURAND._generator[] = nothing
        CUDNN._handle[] = C_NULL
        CUTENSOR._handle[] = C_NULL
    end
    push!(CUDAnative.device!_listeners, callback)

    # a device might be active already
    existing_ctx = CUDAdrv.CuCurrentContext()
    if existing_ctx !== nothing
        active_context[] = existing_ctx
    end

    __init_memory__()
end

end # module
