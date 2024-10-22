# on-device sparse array functionality

using SparseArrays

# NOTE: this functionality is currently very bare-bones, only defining the array types
#       without any device-compatible sparse array functionality


# core types

export CuSparseDeviceVector, CuSparseDeviceMatrixCSC, CuSparseDeviceMatrixCSR,
       CuSparseDeviceMatrixBSR, CuSparseDeviceMatrixCOO

struct CuSparseDeviceVector{Tv,Ti,A} <: AbstractSparseVector{Tv,Ti}
    iPtr::CuDeviceVector{Ti,A,Ti}
    nzVal::CuDeviceVector{Tv,A,Ti}
    len::Int
    nnz::Ti
end

Base.length(g::CuSparseDeviceVector) = prod(g.dims)
Base.size(g::CuSparseDeviceVector) = (g.len,)
SparseArrays.nnz(g::CuSparseDeviceVector) = g.nnz

struct CuSparseDeviceMatrixCSC{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    colPtr::CuDeviceVector{Ti,A,Ti}
    rowVal::CuDeviceVector{Ti,A,Ti}
    nzVal::CuDeviceVector{Tv,A,Ti}
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixCSC) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSC) = g.dims
SparseArrays.nnz(g::CuSparseDeviceMatrixCSC) = g.nnz

struct CuSparseDeviceMatrixCSR{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    rowPtr::CuDeviceVector{Ti,A,Ti}
    colVal::CuDeviceVector{Ti,A,Ti}
    nzVal::CuDeviceVector{Tv,A,Ti}
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixCSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSR) = g.dims
SparseArrays.nnz(g::CuSparseDeviceMatrixCSR) = g.nnz

struct CuSparseDeviceMatrixBSR{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    rowPtr::CuDeviceVector{Ti,A,Ti}
    colVal::CuDeviceVector{Ti,A,Ti}
    nzVal::CuDeviceVector{Tv,A,Ti}
    dims::NTuple{2,Int}
    blockDim::Ti
    dir::Char
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixBSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixBSR) = g.dims
SparseArrays.nnz(g::CuSparseDeviceMatrixBSR) = g.nnz

struct CuSparseDeviceMatrixCOO{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    rowInd::CuDeviceVector{Ti,A,Ti}
    colInd::CuDeviceVector{Ti,A,Ti}
    nzVal::CuDeviceVector{Tv,A,Ti}
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixCOO) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCOO) = g.dims
SparseArrays.nnz(g::CuSparseDeviceMatrixCOO) = g.nnz

struct CuSparseDeviceArrayCSR{Tv, Ti, N, M, A} <: AbstractSparseArray{Tv, Ti, N}
    rowPtr::CuDeviceArray{Ti, M, A} 
    colVal::CuDeviceArray{Ti, M, A} 
    nzVal::CuDeviceArray{Tv, M, A} 
    dims::NTuple{N, Int}
    nnz::Ti
end

function CuSparseDeviceArrayCSR{Tv, Ti, N, A}(rowPtr::CuArray{<:Integer, M}, colVal::CuArray{<:Integer, M}, nzVal::CuArray{Tv, M}, dims::NTuple{N,<:Integer}) where {Tv, Ti<:Integer, M, N, A}
    @assert M == N - 1 "CuSparseDeviceArrayCSR requires ndims(rowPtr) == ndims(colVal) == ndims(nzVal) == length(dims) - 1"
    CuSparseDeviceArrayCSR{Tv, Ti, N, M, A}(rowPtr, colVal, nzVal, dims, length(nzVal))
end

Base.length(g::CuSparseDeviceArrayCSR) = prod(g.dims)
Base.size(g::CuSparseDeviceArrayCSR) = g.dims
SparseArrays.nnz(g::CuSparseDeviceArrayCSR) = g.nnz

# input/output

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceVector)
    println(io, "$(length(A))-element device sparse vector at:")
    println(io, "  iPtr: $(A.iPtr)")
    print(io,   "  nzVal: $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceMatrixCSR)
    println(io, "$(length(A))-element device sparse matrix CSR at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colVal: $(A.colVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceMatrixCSC)
    println(io, "$(length(A))-element device sparse matrix CSC at:")
    println(io, "  colPtr: $(A.colPtr)")
    println(io, "  rowVal: $(A.rowVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceMatrixBSR)
    println(io, "$(length(A))-element device sparse matrix BSR at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colVal: $(A.colVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceMatrixCOO)
    println(io, "$(length(A))-element device sparse matrix COO at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colInd: $(A.colInd)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceArrayCSR)
    println(io, "$(length(A))-element device sparse array CSR at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colVal: $(A.colVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end
