module CUSOLVER

using ..CuArrays
using ..CuArrays: active_context, _getindex, unsafe_free!

using ..CUBLAS: cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasDiagType_t
using ..CUSPARSE: cusparseMatDescr_t

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

const libcusolver = if Sys.iswindows()
    # no ccall by soname, we need the filename
    # NOTE: we discover the full path here, while only the wordsize and toolkit versions
    #       would have been enough to construct "cusolver64_10.dll"
    toolkit = find_toolkit()
    path = find_cuda_library("cusolver", toolkit)
    if path === nothing
        error("Could not find libcusolver")
    end
    basename(path)
else
    # ccall by soname; CuArrays.__init__ will have populated Libdl.DL_LOAD_PATH
    "libcusolver"
end

# core library
include("libcusolver_common.jl")
include("error.jl")
include("libcusolver.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("linalg.jl")

const _dense_handles = Dict{CuContext,cusolverDnHandle_t}()
const _dense_handle = Ref{cusolverDnHandle_t}(C_NULL)
const _sparse_handles = Dict{CuContext,cusolverSpHandle_t}()
const _sparse_handle = Ref{cusolverSpHandle_t}(C_NULL)

function dense_handle()
    if _dense_handle[] == C_NULL
        CUDAnative.maybe_initialize("CUSOLVER")
        _dense_handle[] = get!(_dense_handles, active_context[]) do
            context = active_context[]
            handle = cusolverDnCreate()
            atexit(()->CUDAdrv.isvalid(context) && cusolverDnDestroy(handle))
            handle
        end
    end
    return _dense_handle[]
end

function sparse_handle()
    if _sparse_handle[] == C_NULL
        CUDAnative.maybe_initialize("CUSOLVER")
        _sparse_handle[] = get!(_sparse_handles, active_context[]) do
            context = active_context[]
            handle = cusolverSpCreate()
            atexit(()->CUDAdrv.isvalid(context) && cusolverSpDestroy(handle))
            handle
        end
    end
    return _sparse_handle[]
end

end
