module CuArrays

using CUDAapi, CUDAdrv, CUDAnative

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu

import LinearAlgebra

using Adapt

using Requires

## discovery

let
    toolkit = find_toolkit()

    # required libraries that are part of the CUDA toolkit
    for name in ("cublas", "cusparse", "cusolver", "cufft", "curand")
        lib = Symbol("lib$name")
        path = find_cuda_library(name, toolkit)
        if path === nothing
            error("Could not find library '$name' (it should be part of the CUDA toolkit)")
        end
        Base.include_dependency(path)
        @eval global const $lib = $path
    end

    # optional libraries
    for name in ("cudnn", "cutensor")
        lib = Symbol("lib$name")
        path = find_cuda_library(name, toolkit)
        mod = uppercase(name)
        if path !== nothing
            Base.include_dependency(path)
            @debug "Found $mod at $path"
        else
            @warn "You installation does not provide $lib, CuArrays.$mod will be unavailable"
        end

        # provide a global constant that returns the path to the library,
        # or nothing if the library is not available (for use in conditional expressions)
        @eval global const $lib = $path

        # provide a macro that either returns the path to the library,
        # or a run-time error if the library is not available (for use in ccall expressions)
        exception = :(error($"Your installation does not provide $lib, CuArrays.$(uppercase(name)) is unavailable"))
        @eval macro $lib() $lib === nothing ? $(QuoteNode(exception)) : $lib end

        # provide a function for external use (a la CUDAapi.has_cuda)
        fn = Symbol("has_$name")
        @eval (export $fn; $fn() = $lib !== nothing)
    end
end


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
libcutensor !== nothing && include("tensor/CUTENSOR.jl")

include("nnlib.jl")

include("deprecated.jl")


## initialization
function __init__()
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
        isdefined(CuArrays, :CUTENSOR) && (CUTENSOR._handle[] = C_NULL)
    end
    push!(CUDAnative.device!_listeners, callback)

    # a device might be active already
    existing_ctx = CUDAdrv.CuCurrentContext()
    if existing_ctx !== nothing
        active_context[] = existing_ctx
    end

    __init_memory__()
    __init_pool_()
end

end # module
