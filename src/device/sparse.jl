# on-device sparse array functionality

using SparseArrays

# NOTE: this functionality is currently very bare-bones, only defining the array types
#       without any device-compatible sparse array functionality


# core types

export CuSparseDeviceVector, CuSparseDeviceMatrixCSC, CuSparseDeviceMatrixCSR,
       CuSparseDeviceMatrixBSR, CuSparseDeviceMatrixCOO

mutable struct CuSparseDeviceVector{Tv,Ti} <: AbstractSparseVector{Tv,Ti}
    iPtr::CuDeviceVector{Ti, AS.Global}
    nzVal::CuDeviceVector{Tv, AS.Global}
    dims::NTuple{2,Int}
    nnz::Int
end

Base.length(g::CuSparseDeviceVector) = prod(g.dims)
Base.size(g::CuSparseDeviceVector) = g.dims
Base.ndims(g::CuSparseDeviceVector) = 1

mutable struct CuSparseDeviceMatrixCSC{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    colPtr::CuDeviceVector{Ti, AS.Global}
    rowVal::CuDeviceVector{Ti, AS.Global}
    nzVal::CuDeviceVector{Tv, AS.Global}
    dims::NTuple{2,Int}
    nnz::Int
end

Base.length(g::CuSparseDeviceMatrixCSC) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSC) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCSC) = 2

struct CuSparseDeviceMatrixCSR{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    rowPtr::CuDeviceVector{Ti, AS.Global}
    colVal::CuDeviceVector{Ti, AS.Global}
    nzVal::CuDeviceVector{Tv, AS.Global}
    dims::NTuple{2, Int}
    nnz::Int
end

Base.length(g::CuSparseDeviceMatrixCSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSR) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCSR) = 2

mutable struct CuSparseDeviceMatrixBSR{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    rowPtr::CuDeviceVector{Ti}
    colVal::CuDeviceVector{Ti}
    nzVal::CuDeviceVector{Tv}
    dims::NTuple{2,Int}
    blockDim::Int
    dir::Char
    nnz::Int
end

Base.length(g::CuSparseDeviceMatrixBSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixBSR) = g.dims
Base.ndims(g::CuSparseDeviceMatrixBSR) = 2

struct CuSparseDeviceMatrixCOO{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    rowInd::CuDeviceVector{Ti}
    colInd::CuDeviceVector{Ti}
    nzVal::CuDeviceVector{Tv}
    dims::NTuple{2,Int}
    nnz::Int
end

Base.length(g::CuSparseDeviceMatrixCOO) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCOO) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCOO) = 2


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
