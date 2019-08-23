module CUSOLVER

import CUDAapi

import CUDAdrv: CUDAdrv, CuContext, CuStream_t, CuPtr, PtrOrCuPtr, CU_NULL

import CUDAnative

using ..CuArrays
using ..CuArrays: libcusolver, active_context, _getindex, unsafe_free!

using LinearAlgebra
using SparseArrays

import Base.one
import Base.zero
import CuArrays.CUSPARSE.CuSparseMatrixCSR
import CuArrays.CUSPARSE.CuSparseMatrixCSC
import CuArrays.CUSPARSE.cusparseMatDescr_t

include("libcusolver_types.jl")
include("error.jl")

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

include("libcusolver.jl")
include("sparse.jl")
include("dense.jl")
include("highlevel.jl")

version() = VersionNumber(cusolverGetProperty(CUDAapi.MAJOR_VERSION),
                          cusolverGetProperty(CUDAapi.MINOR_VERSION),
                          cusolverGetProperty(CUDAapi.PATCH_LEVEL))

end
