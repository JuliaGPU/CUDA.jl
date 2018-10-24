module CUSOLVER

import CUDAdrv
using CUDAdrv: CuContext, CuDevice
using CUDAnative

using ..CuArrays
const CuStream_t = Ptr{Nothing}

using ..CuArrays: libcusolver, active_context, _getindex

using LinearAlgebra
using SparseArrays 

import Base.one
import Base.zero
import CuArrays.CUSPARSE.CuSparseMatrixCSR
import CuArrays.CUSPARSE.CuSparseMatrixCSC
import CuArrays.CUSPARSE.cusparseMatDescr_t

include("libcusolver_types.jl")

const _dense_handles = Dict{CuContext,cusolverDnHandle_t}()
const _dense_handle = Ref{cusolverDnHandle_t}(C_NULL)

const _dense_handles = Dict{CuContext,cusolverDnHandle_t}()
const _dense_handle = Ref{cusolverDnHandle_t}(C_NULL)
const _sparse_handles = Dict{CuContext,cusolverSpHandle_t}()
const _sparse_handle = Ref{cusolverSpHandle_t}(C_NULL)

function dense_handle()
    if _dense_handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
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
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _sparse_handle[] = get!(_sparse_handles, active_context[]) do
            context = active_context[]
            handle = cusolverSpCreate()
            atexit(()->CUDAdrv.isvalid(context) && cusolverSpDestroy(handle))
            handle
        end
    end
    return _sparse_handle[]
end

include("error.jl")
include("libcusolver.jl")
include("sparse.jl")
include("dense.jl")
include("highlevel.jl")

end
