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

if VERSION >= v"1.3.0-DEV.35"
    using Base: inferencebarrier
else
    inferencebarrier(@nospecialize(x)) = Ref{Any}(x)[]
end

function __init__()
    if ccall(:jl_generating_output, Cint, ()) == 1
        # don't initialize when we, or any package that depends on us, is precompiling.
        # this makes it possible to precompile on systems without CUDA,
        # at the expense of using the packages in global scope.
        return
    end

    silent = parse(Bool, get(ENV, "CUDA_INIT_SILENT", "false"))

    # discover libraries
    # NOTE: we can't just ccall by soname because that's not possible on Windows,
    #       and CUDA libraries there contain a version number in the filename.
    toolkit = find_toolkit()
    for name in ("cublas", "cusparse", "cusolver", "cufft", "curand", "cudnn", "cutensor")
        mod = getfield(CuArrays, Symbol(uppercase(name)))
        lib = Symbol("lib$name")
        path = find_cuda_library(name, toolkit)
        @eval mod const $lib = $path
    end

    try
        # barrier to avoid compiling `ccall`s to unavailable libraries
        inferencebarrier(__hidden_init__)()
        @eval functional() = true
    catch ex
        # don't actually fail to keep the package loadable
        silent || @error """CuArrays.jl failed to initialize; the package will not be functional.
                            To silence this message, import with ENV["CUDA_INIT_SILENT"]=true,
                            and be sure to inspect the value of CuArrays.functional().""" exception=(ex, catch_backtrace())
        @eval functional() = false
    end
end

export has_cudnn, has_cutensor
has_cudnn() = CUDNN.libcudnn !== nothing
has_cutensor() = CUTENSOR.libcutensor !== nothing

function __hidden_init__()
    # package dependencies
    CUDAdrv.functional() || error("CUDAdrv.jl is not functional")
    CUDAnative.functional() || error("CUDAnative.jl is not functional")

    # library dependencies
    CUBLAS.version()
    CUSPARSE.version()
    CUSOLVER.version()
    CUFFT.version()
    CURAND.version()
    has_cudnn() || @warn "Could not find libcudnn, CuArrays.CUDNN will be unavailable."
    has_cutensor() || @warn "Could not find libcutensor, CuArrays.CUTENSOR will be unavailable."

    # library compatibility
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
