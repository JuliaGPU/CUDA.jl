module CUSPARSE

import CUDAdrv: CUDAdrv, CuContext, CuStream_t

using ..CuArrays
using ..CuArrays: libcusparse, active_context

using SparseArrays
using LinearAlgebra

import Base.one
import Base.zero

const SparseChar = Char
import Base.one
import Base.zero

export CuSparseMatrixCSC, CuSparseMatrixCSR,
       CuSparseMatrixHYB, CuSparseMatrixBSR,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseVector

include("libcusparse_types.jl")
include("error.jl")
include("libcusparse.jl")

const _handles = Dict{CuContext,cusparseHandle_t}()
const _handle = Ref{cusparseHandle_t}()

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cusparseCreate()
            atexit(()->CUDAdrv.isvalid(context) && cusparseDestroy(handle))
            handle
        end
    end

    return _handle[]
end

include("array.jl")
include("util.jl")
include("wrappers.jl")
include("highlevel.jl")

end
