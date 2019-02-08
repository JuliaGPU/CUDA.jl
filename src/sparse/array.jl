# custom extension of CuArray in CUDArt for sparse vectors/matrices
# using CSC format for interop with Julia's native sparse functionality

import Base: length, size, ndims, eltype, similar, pointer, stride,
    copy, convert, reinterpret, show, summary, copyto!, get!, fill!, collect
import LinearAlgebra: BlasFloat, Hermitian, HermOrSym, issymmetric,
    ishermitian, istriu, istril, Symmetric, UpperTriangular, LowerTriangular
import SparseArrays: sparse, SparseMatrixCSC

abstract type AbstractCuSparseArray{Tv, N} <: AbstractSparseArray{Tv, Cint, N} end
const AbstractCuSparseVector{Tv} = AbstractCuSparseArray{Tv,1}
const AbstractCuSparseMatrix{Tv} = AbstractCuSparseArray{Tv,2}

mutable struct CuSparseVector{Tv} <: AbstractCuSparseVector{Tv}
    iPtr::CuVector{Cint}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Cint

    function CuSparseVector{Tv}(iPtr::CuVector{Cint}, nzVal::CuVector{Tv}, dims::Int, nnz::Cint) where Tv
        new(iPtr,nzVal,(dims,1),nnz)
    end
end

function CuArrays.unsafe_free!(xs::CuSparseVector)
    unsafe_free!(xs.iPtr)
    unsafe_free!(xs.nzVal)
    return
end

mutable struct CuSparseMatrixCSC{Tv} <: AbstractCuSparseMatrix{Tv}
    colPtr::CuVector{Cint}
    rowVal::CuVector{Cint}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Cint

    function CuSparseMatrixCSC{Tv}(colPtr::CuVector{Cint}, rowVal::CuVector{Cint}, nzVal::CuVector{Tv}, dims::NTuple{2,Int}, nnz::Cint) where Tv
        new(colPtr,rowVal,nzVal,dims,nnz)
    end
end

function CuSparseMatrixCSC!(xs::CuSparseVector)
    unsafe_free!(xs.colPtr)
    unsafe_free!(xs.rowVal)
    unsafe_free!(xs.nzVal)
    return
end

"""
Container to hold sparse matrices in compressed sparse row (CSR) format on the
GPU.

**Note**: Most CUSPARSE operations work with CSR formatted matrices, rather
than CSC.
"""
mutable struct CuSparseMatrixCSR{Tv} <: AbstractCuSparseMatrix{Tv}
    rowPtr::CuVector{Cint}
    colVal::CuVector{Cint}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Cint

    function CuSparseMatrixCSR{Tv}(rowPtr::CuVector{Cint}, colVal::CuVector{Cint}, nzVal::CuVector{Tv}, dims::NTuple{2,Int}, nnz::Cint) where Tv
        new(rowPtr,colVal,nzVal,dims,nnz)
    end
end

function CuSparseMatrixCSR!(xs::CuSparseVector)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(xs.nzVal)
    return
end

"""
Container to hold sparse matrices in block compressed sparse row (BSR) format on
the GPU. BSR format is also used in Intel MKL, and is suited to matrices that are
"block" sparse - rare blocks of non-sparse regions.
"""
mutable struct CuSparseMatrixBSR{Tv} <: AbstractCuSparseMatrix{Tv}
    rowPtr::CuVector{Cint}
    colVal::CuVector{Cint}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    blockDim::Cint
    dir::SparseChar
    nnz::Cint

    function CuSparseMatrixBSR{Tv}(rowPtr::CuVector{Cint}, colVal::CuVector{Cint}, nzVal::CuVector{Tv}, dims::NTuple{2,Int},blockDim::Cint, dir::SparseChar, nnz::Cint) where Tv
        new(rowPtr,colVal,nzVal,dims,blockDim,dir,nnz)
    end
end

function CuSparseMatrixBSR!(xs::CuSparseVector)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(xs.nzVal)
    return
end

"""
Container to hold sparse matrices in NVIDIA's hybrid (HYB) format on the GPU.
HYB format is an opaque struct, which can be converted to/from using
CUSPARSE routines.
"""
mutable struct CuSparseMatrixHYB{Tv} <: AbstractCuSparseMatrix{Tv}
    Mat::cusparseHybMat_t
    dims::NTuple{2,Int}
    nnz::Cint

    function CuSparseMatrixHYB{Tv}(Mat::cusparseHybMat_t, dims::NTuple{2,Int}, nnz::Cint) where Tv
        new(Mat,dims,nnz)
    end
end

"""
Utility union type of [`CuSparseMatrixCSC`](@ref), [`CuSparseMatrixCSR`](@ref),
and `Hermitian` and `Symmetric` versions of these two containers. A function accepting
this type can make use of performance improvements by only indexing one triangle of the
matrix if it is guaranteed to be hermitian/symmetric.
"""
const CompressedSparse{T} = Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},HermOrSym{T,CuSparseMatrixCSC{T}},HermOrSym{T,CuSparseMatrixCSR{T}}}

"""
Utility union type of [`CuSparseMatrixCSC`](@ref), [`CuSparseMatrixCSR`](@ref),
[`CuSparseMatrixBSR`](@ref), and [`CuSparseMatrixHYB`](@ref).
"""
const CuSparseMatrix{T} = Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T}, CuSparseMatrixBSR{T}, CuSparseMatrixHYB{T}}

Hermitian{T}(Mat::CuSparseMatrix{T}) where T = Hermitian{T,typeof(Mat)}(Mat,'U')

length(g::CuSparseVector) = prod(g.dims)
size(g::CuSparseVector) = g.dims
ndims(g::CuSparseVector) = 1
length(g::CuSparseMatrix) = prod(g.dims)
size(g::CuSparseMatrix) = g.dims
ndims(g::CuSparseMatrix) = 2

function size(g::CuSparseVector, d::Integer)
    if d == 1
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

function size(g::CuSparseMatrix, d::Integer)
    if d in [1, 2]
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

issymmetric(M::Union{CuSparseMatrixCSC,CuSparseMatrixCSR})= false
ishermitian(M::Union{CuSparseMatrixCSC,CuSparseMatrixCSR}) where T = false
issymmetric(M::Symmetric{CuSparseMatrixCSC})= true
ishermitian(M::Hermitian{CuSparseMatrixCSC}) = true

for mat_type in [:CuSparseMatrixCSC, :CuSparseMatrixCSR, :CuSparseMatrixBSR, :CuSparseMatrixHYB]
    @eval begin
        istriu(M::UpperTriangular{$mat_type}) = true
        istril(M::UpperTriangular{$mat_type}) = false
        istriu(M::LowerTriangular{$mat_type}) = false 
        istril(M::LowerTriangular{$mat_type}) = true 
    end
end
eltype(g::CuSparseMatrix{T}) where T = T

function collect(Vec::CuSparseVector)
    SparseVector(Vec.dims[1], collect(Vec.iPtr), collect(Vec.nzVal))
end

function collect(Mat::CuSparseMatrixCSC)
    SparseMatrixCSC(Mat.dims[1], Mat.dims[2], collect(Mat.colPtr), collect(Mat.rowVal), collect(Mat.nzVal))
end
function collect(Mat::CuSparseMatrixCSR)
    rowPtr = collect(Mat.rowPtr)
    colVal = collect(Mat.colVal)
    nzVal = collect(Mat.nzVal)
    #construct Is
    I = similar(colVal)
    counter = 1
    for row = 1 : size(Mat)[1], k = rowPtr[row] : (rowPtr[row+1]-1)
        I[counter] = row
        counter += 1
    end
    return sparse(I,colVal,nzVal,Mat.dims[1],Mat.dims[2])
end

summary(g::CuSparseMatrix) = string(g)
summary(g::CuSparseVector) = string(g)

CuSparseVector(iPtr::Vector{Ti}, nzVal::Vector{T}, dims::Int) where {T<:BlasFloat, Ti<:Integer} = CuSparseVector{T}(CuArray(convert(Vector{Cint},iPtr)), CuArray(nzVal), dims, convert(Cint,length(nzVal)))
CuSparseVector(iPtr::CuArray{Ti}, nzVal::CuArray{T}, dims::Int) where {T<:BlasFloat, Ti<:Integer} = CuSparseVector{T}(iPtr, nzVal, dims, convert(Cint,length(nzVal)))

CuSparseMatrixCSC(colPtr::Vector{Ti}, rowVal::Vector{Ti}, nzVal::Vector{T}, dims::NTuple{2,Int}) where {T<:BlasFloat,Ti<:Integer} = CuSparseMatrixCSC{T}(CuArray(convert(Vector{Cint},colPtr)), CuArray(convert(Vector{Cint},rowVal)), CuArray(nzVal), dims, convert(Cint,length(nzVal)))
CuSparseMatrixCSC(colPtr::CuArray{Ti}, rowVal::CuArray{Ti}, nzVal::CuArray{T}, dims::NTuple{2,Int}) where {T<:BlasFloat,Ti<:Integer} = CuSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)))
CuSparseMatrixCSC(colPtr::CuArray{Ti}, rowVal::CuArray{Ti}, nzVal::CuArray{T}, nnz, dims::NTuple{2,Int}) where {T<:BlasFloat,Ti<:Integer} = CuSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, nnz)

CuSparseMatrixCSR(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, dims::NTuple{2,Int}) where T = CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)))
CuSparseMatrixCSR(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, nnz, dims::NTuple{2,Int}) where T = CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, nnz)

CuSparseMatrixBSR(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, blockDim, dir, nnz, dims::NTuple{2,Int}) where T = CuSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)

CuSparseVector(Vec::SparseVector)    = CuSparseVector(Vec.nzind, Vec.nzval, size(Vec)[1])
CuSparseMatrixCSC(Vec::SparseVector)    = CuSparseMatrixCSC([1], Vec.nzind, Vec.nzval, size(Vec))
CuSparseVector(Mat::SparseMatrixCSC) = size(Mat,2) == 1 ? CuSparseVector(Mat.rowval, Mat.nzval, size(Mat)[1]) : throw(ArgumentError())
CuSparseMatrixCSC(Mat::SparseMatrixCSC) = CuSparseMatrixCSC(Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))
CuSparseMatrixCSR(Mat::SparseMatrixCSC) = switch2csr(CuSparseMatrixCSC(Mat))

similar(Vec::CuSparseVector) = CuSparseVector(copy(Vec.iPtr), similar(Vec.nzVal), Vec.dims[1])
similar(Mat::CuSparseMatrixCSC) = CuSparseMatrixCSC(copy(Mat.colPtr), copy(Mat.rowVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CuSparseMatrixCSR) = CuSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CuSparseMatrixBSR) = CuSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.blockDim, Mat.dir, Mat.nnz, Mat.dims)

function copyto!(dst::CuSparseVector, src::CuSparseVector)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Vector size"))
    end
    copyto!(dst.iPtr, src.iPtr)
    copyto!(dst.nzVal, src.nzVal)
    dst.nnz = src.nnz
    dst
end

function copyto!(dst::CuSparseMatrixCSC, src::CuSparseMatrixCSC)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.colPtr, src.colPtr)
    copyto!(dst.rowVal, src.rowVal)
    copyto!(dst.nzVal, src.nzVal)
    dst.nnz = src.nnz
    dst
end

function copyto!(dst::CuSparseMatrixCSR, src::CuSparseMatrixCSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(dst.nzVal, src.nzVal)
    dst.nnz = src.nnz
    dst
end

function copyto!(dst::CuSparseMatrixBSR, src::CuSparseMatrixBSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(dst.nzVal, src.nzVal)
    dst.dir = src.dir
    dst.nnz = src.nnz
    dst
end

function copyto!(dst::CuSparseMatrixHYB, src::CuSparseMatrixHYB)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    dst.Mat = src.Mat
    dst.nnz = src.nnz
    dst
end

copy(Vec::CuSparseVector) = copyto!(similar(Vec),Vec)
copy(Mat::CuSparseMatrixCSC) = copyto!(similar(Mat),Mat)
copy(Mat::CuSparseMatrixCSR) = copyto!(similar(Mat),Mat)
copy(Mat::CuSparseMatrixBSR) = copyto!(similar(Mat),Mat)
