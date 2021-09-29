# on-device sparse array functionality

using SparseArrays

# NOTE: this functionality is currently very bare-bones, only defining the array types
#       without any device-compatible sparse array functionality


# core types

export CuSparseDeviceVector, CuSparseDeviceMatrixCSC, CuSparseDeviceMatrixCSR,
       CuSparseDeviceMatrixBSR, CuSparseDeviceMatrixCOO

struct CuSparseDeviceVector{Tv,Ti, A} <: AbstractSparseVector{Tv,Ti}
    iPtr::CuDeviceVector{Ti, A}
    nzVal::CuDeviceVector{Tv, A}
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::CuSparseDeviceVector) = prod(g.dims)
Base.size(g::CuSparseDeviceVector) = g.dims
Base.ndims(g::CuSparseDeviceVector) = 1

struct CuSparseDeviceMatrixCSC{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    colPtr::CuDeviceVector{Ti, A}
    rowVal::CuDeviceVector{Ti, A}
    nzVal::CuDeviceVector{Tv, A}
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixCSC) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSC) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCSC) = 2

struct CuSparseDeviceMatrixCSR{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    rowPtr::CuDeviceVector{Ti, A}
    colVal::CuDeviceVector{Ti, A}
    nzVal::CuDeviceVector{Tv, A}
    dims::NTuple{2, Int}
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixCSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSR) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCSR) = 2

struct CuSparseDeviceMatrixBSR{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    rowPtr::CuDeviceVector{Ti, A}
    colVal::CuDeviceVector{Ti, A}
    nzVal::CuDeviceVector{Tv, A}
    dims::NTuple{2,Int}
    blockDim::Ti
    dir::Char
    nnz::Ti
end

Base.length(g::CuSparseDeviceMatrixBSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixBSR) = g.dims
Base.ndims(g::CuSparseDeviceMatrixBSR) = 2

struct CuSparseDeviceMatrixCOO{Tv,Ti,A} <: AbstractSparseMatrix{Tv,Ti}
    rowInd::CuDeviceVector{Ti, A}
    colInd::CuDeviceVector{Ti, A}
    nzVal::CuDeviceVector{Tv, A}
    dims::NTuple{2,Int}
    nnz::Ti
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
