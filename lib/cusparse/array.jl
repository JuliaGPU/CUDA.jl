# custom extension of CuArray in CUDArt for sparse vectors/matrices
# using CSC format for interop with Julia's native sparse functionality

export CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR, CuSparseMatrixCOO,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseArrayCSR,
       CuSparseVector,
       CuSparseVecOrMat

using LinearAlgebra: BlasFloat
using SparseArrays: nonzeroinds, dimlub

abstract type AbstractCuSparseArray{Tv, Ti, N} <: AbstractSparseArray{Tv, Ti, N} end
const AbstractCuSparseVector{Tv, Ti} = AbstractCuSparseArray{Tv, Ti, 1}
const AbstractCuSparseMatrix{Tv, Ti} = AbstractCuSparseArray{Tv, Ti, 2}

Base.convert(T::Type{<:AbstractCuSparseArray}, m::AbstractArray) = m isa T ? m : T(m)

mutable struct CuSparseVector{Tv, Ti} <: AbstractCuSparseVector{Tv, Ti}
    iPtr::CuVector{Ti}
    nzVal::CuVector{Tv}
    len::Int
    nnz::Ti

    function CuSparseVector{Tv, Ti}(iPtr::CuVector{<:Integer}, nzVal::CuVector,
                                    len::Integer) where {Tv, Ti <: Integer}
        new{Tv, Ti}(iPtr, nzVal, len, length(nzVal))
    end
end

function CUDA.unsafe_free!(xs::CuSparseVector)
    unsafe_free!(nonzeroinds(xs))
    unsafe_free!(nonzeros(xs))
    return
end

mutable struct CuSparseMatrixCSC{Tv, Ti} <: AbstractCuSparseMatrix{Tv, Ti}
    colPtr::CuVector{Ti}
    rowVal::CuVector{Ti}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Ti

    function CuSparseMatrixCSC{Tv, Ti}(colPtr::CuVector{<:Integer}, rowVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer}) where {Tv, Ti <: Integer}
        new{Tv, Ti}(colPtr, rowVal, nzVal, dims, length(nzVal))
    end
end

CuSparseMatrixCSC(A::CuSparseMatrixCSC) = A

function CUDA.unsafe_free!(xs::CuSparseMatrixCSC)
    unsafe_free!(xs.colPtr)
    unsafe_free!(rowvals(xs))
    unsafe_free!(nonzeros(xs))
    return
end

"""
    CuSparseMatrixCSR{Tv, Ti} <: AbstractCuSparseMatrix{Tv, Ti}

Container to hold sparse matrices in compressed sparse row (CSR) format on the
GPU.

!!! note
    Most CUSPARSE operations work with CSR formatted matrices, rather
    than CSC.

!!! compat "CUDA 11"
    Support of indices type rather than `Cint` (`Int32`) requires at least CUDA 11.
"""
mutable struct CuSparseMatrixCSR{Tv, Ti} <: AbstractCuSparseMatrix{Tv, Ti}
    rowPtr::CuVector{Ti}
    colVal::CuVector{Ti}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Ti

    function CuSparseMatrixCSR{Tv, Ti}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer}) where {Tv, Ti<:Integer}
        new{Tv, Ti}(rowPtr, colVal, nzVal, dims, length(nzVal))
    end
end

CuSparseMatrixCSR(A::CuSparseMatrixCSR) = A

function CUDA.unsafe_free!(xs::CuSparseMatrixCSR)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(nonzeros(xs))
    return
end

"""
Container to hold sparse matrices in block compressed sparse row (BSR) format on
the GPU. BSR format is also used in Intel MKL, and is suited to matrices that are
"block" sparse - rare blocks of non-sparse regions.
"""
mutable struct CuSparseMatrixBSR{Tv, Ti} <: AbstractCuSparseMatrix{Tv, Ti}
    rowPtr::CuVector{Ti}
    colVal::CuVector{Ti}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    blockDim::Ti
    dir::SparseChar
    nnzb::Ti

    function CuSparseMatrixBSR{Tv, Ti}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer},
                                   blockDim::Integer, dir::SparseChar, nnz::Integer) where {Tv, Ti<:Integer}
        new{Tv, Ti}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)
    end
end

CuSparseMatrixBSR(A::CuSparseMatrixBSR) = A

function CUDA.unsafe_free!(xs::CuSparseMatrixBSR)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(nonzeros(xs))
    return
end

"""
Container to hold sparse matrices in coordinate (COO) format on the GPU. COO
format is mainly useful to initially construct sparse matrices, afterwards
switch to [`CuSparseMatrixCSR`](@ref) for more functionality.
"""
mutable struct CuSparseMatrixCOO{Tv, Ti} <: AbstractCuSparseMatrix{Tv, Ti}
    rowInd::CuVector{Ti}
    colInd::CuVector{Ti}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Ti

    function CuSparseMatrixCOO{Tv, Ti}(rowInd::CuVector{<:Integer}, colInd::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer}=(dimlub(rowInd),dimlub(colInd)),
                                   nnz::Integer=length(nzVal)) where {Tv, Ti}
        new{Tv, Ti}(rowInd,colInd,nzVal,dims,nnz)
    end
end

CuSparseMatrixCOO(A::CuSparseMatrixCOO) = A

mutable struct CuSparseArrayCSR{Tv, Ti, N} <: AbstractCuSparseArray{Tv, Ti, N}
    rowPtr::CuArray{Ti}
    colVal::CuArray{Ti}
    nzVal::CuArray{Tv}
    dims::NTuple{N,Int}
    nnz::Ti

    function CuSparseArrayCSR{Tv, Ti, N}(rowPtr::CuArray{<:Integer, M}, colVal::CuArray{<:Integer, M}, nzVal::CuArray{Tv, M}, dims::NTuple{N,<:Integer}) where {Tv, Ti<:Integer, M, N}
        @assert M == N - 1 "CuSparseArrayCSR requires ndims(rowPtr) == ndims(colVal) == ndims(nzVal) == length(dims) - 1"
        new{Tv, Ti, N}(rowPtr, colVal, nzVal, dims, length(nzVal))
    end
end

CuSparseArrayCSR(A::CuSparseArrayCSR) = A

function CUDA.unsafe_free!(xs::CuSparseArrayCSR)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(nonzeros(xs))
    return
end

# broadcast over batch-dim if batchsize==1
ptrstride(A::CuSparseArrayCSR) = size(A.rowPtr, 2) > 1 ? stride(A.rowPtr, 2) : 0
valstride(A::CuSparseArrayCSR) = size(A.nzVal, 2) > 1 ? stride(A.nzVal, 2) : 0

"""
Utility union type of [`CuSparseMatrixCSC`](@ref), [`CuSparseMatrixCSR`](@ref),
[`CuSparseMatrixBSR`](@ref), [`CuSparseMatrixCOO`](@ref).
"""
const CuSparseMatrix{Tv, Ti} = Union{
    CuSparseMatrixCSC{Tv, Ti},
    CuSparseMatrixCSR{Tv, Ti},
    CuSparseMatrixBSR{Tv, Ti},
    CuSparseMatrixCOO{Tv, Ti}
}

const CuSparseVecOrMat = Union{CuSparseVector,CuSparseMatrix}

# NOTE: we use Cint as default Ti on CUDA instead of Int to provide
# maximum compatiblity to old CUSPARSE APIs
function CuSparseVector{Tv}(iPtr::CuVector{<:Integer}, nzVal::CuVector, len::Integer) where {Tv}
    CuSparseVector{Tv, Cint}(convert(CuVector{Cint}, iPtr), nzVal, len)
end

function CuSparseMatrixCSC{Tv}(colPtr::CuVector{<:Integer}, rowVal::CuVector{<:Integer},
                               nzVal::CuVector, dims::NTuple{2,<:Integer}) where {Tv}
    CuSparseMatrixCSC{Tv, Cint}(colPtr, rowVal, nzVal, dims)
end

function CuSparseMatrixCSR{Tv}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                               nzVal::CuVector, dims::NTuple{2,<:Integer}) where {Tv}
    CuSparseMatrixCSR{Tv, Cint}(rowPtr, colVal, nzVal, dims)
end

function CuSparseMatrixBSR{Tv}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                               nzVal::CuVector, dims::NTuple{2,<:Integer},
                               blockDim::Integer, dir::SparseChar, nnz::Integer) where {Tv}
    CuSparseMatrixBSR{Tv, Cint}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)
end

function CuSparseMatrixCOO{Tv}(rowInd::CuVector{<:Integer}, colInd::CuVector{<:Integer},
                               nzVal::CuVector, dims::NTuple{2,<:Integer}=(dimlub(rowInd),dimlub(colInd)),
                               nnz::Integer=length(nzVal)) where {Tv}
    CuSparseMatrixCOO{Tv, Cint}(rowInd,colInd,nzVal,dims,nnz)
end

function CuSparseArrayCSR{Tv}(rowPtr::CuArray{<:Integer, M}, colVal::CuArray{<:Integer, M},
                              nzVal::CuArray{Tv, M}, dims::NTuple{N,<:Integer}) where {Tv, M, N}
    CuSparseArrayCSR{Tv, Cint, N}(rowPtr, colVal, nzVal, dims)
end

## convenience constructors
CuSparseVector(iPtr::DenseCuArray{<:Integer}, nzVal::DenseCuArray{T}, len::Integer) where {T} =
    CuSparseVector{T}(iPtr, nzVal, len)

CuSparseMatrixCSC(colPtr::DenseCuArray{<:Integer}, rowVal::DenseCuArray{<:Integer},
                  nzVal::DenseCuArray{T}, dims::NTuple{2,<:Integer}) where {T} =
    CuSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims)

CuSparseMatrixCSR(rowPtr::DenseCuArray, colVal::DenseCuArray, nzVal::DenseCuArray{T}, dims::NTuple{2,<:Integer}) where T =
    CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims)

CuSparseMatrixBSR(rowPtr::DenseCuArray, colVal::DenseCuArray, nzVal::DenseCuArray{T}, blockDim, dir, nnz,
                  dims::NTuple{2,<:Integer}) where T =
    CuSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)

CuSparseMatrixCOO(rowInd::DenseCuArray, colInd::DenseCuArray, nzVal::DenseCuArray{T}, dims::NTuple{2,<:Integer}, nnz::Integer=length(nzVal)) where T =
    CuSparseMatrixCOO{T}(rowInd, colInd, nzVal, dims, nnz)

CuSparseArrayCSR(rowPtr::DenseCuArray, colVal::DenseCuArray, nzVal::DenseCuArray{T}, dims::NTuple{N,<:Integer}) where {T,N} =
    CuSparseArrayCSR{T}(rowPtr, colVal, nzVal, dims)

Base.similar(Vec::CuSparseVector) = CuSparseVector(copy(nonzeroinds(Vec)), similar(nonzeros(Vec)), length(Vec))
Base.similar(Mat::CuSparseMatrixCSC) = CuSparseMatrixCSC(copy(Mat.colPtr), copy(rowvals(Mat)), similar(nonzeros(Mat)), size(Mat))
Base.similar(Mat::CuSparseMatrixCSR) = CuSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), size(Mat))
Base.similar(Mat::CuSparseMatrixBSR) = CuSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), Mat.blockDim, Mat.dir, nnz(Mat), size(Mat))
Base.similar(Mat::CuSparseMatrixCOO) = CuSparseMatrixCOO(copy(Mat.rowInd), copy(Mat.colInd), similar(nonzeros(Mat)), size(Mat), nnz(Mat))

Base.similar(Vec::CuSparseVector, T::Type) = CuSparseVector(copy(nonzeroinds(Vec)), similar(nonzeros(Vec), T), length(Vec))
Base.similar(Mat::CuSparseMatrixCSC, T::Type) = CuSparseMatrixCSC(copy(Mat.colPtr), copy(rowvals(Mat)), similar(nonzeros(Mat), T), size(Mat))
Base.similar(Mat::CuSparseMatrixCSR, T::Type) = CuSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat), T), size(Mat))
Base.similar(Mat::CuSparseMatrixBSR, T::Type) = CuSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat), T), Mat.blockDim, Mat.dir, nnz(Mat), size(Mat))
Base.similar(Mat::CuSparseMatrixCOO, T::Type) = CuSparseMatrixCOO(copy(Mat.rowInd), copy(Mat.colInd), similar(nonzeros(Mat), T), size(Mat), nnz(Mat))

Base.similar(Mat::CuSparseMatrixCSC, T::Type, N::Int, M::Int) =  CuSparseMatrixCSC(CuVector{Int32}(undef, M+1), CuVector{Int32}(undef, nnz(Mat)), CuVector{T}(undef, nnz(Mat)), (N,M))
Base.similar(Mat::CuSparseMatrixCSR, T::Type, N::Int, M::Int) =  CuSparseMatrixCSR(CuVector{Int32}(undef, N+1), CuVector{Int32}(undef, nnz(Mat)), CuVector{T}(undef, nnz(Mat)), (N,M))
Base.similar(Mat::CuSparseMatrixCOO, T::Type, N::Int, M::Int) =  CuSparseMatrixCOO(CuVector{Int32}(undef, nnz(Mat)), CuVector{Int32}(undef, nnz(Mat)), CuVector{T}(undef, nnz(Mat)), (N,M))
Base.similar(Mat::CuSparseArrayCSR) = CuSparseArrayCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), size(Mat))

## array interface

Base.length(g::CuSparseVector) = g.len
Base.size(g::CuSparseVector) = (g.len,)

Base.length(g::CuSparseMatrix) = prod(g.dims)
Base.size(g::CuSparseMatrix) = g.dims

Base.length(g::CuSparseArrayCSR) = prod(g.dims)
Base.size(g::CuSparseArrayCSR) = g.dims

function Base.size(g::CuSparseVector, d::Integer)
    if d == 1
        return g.len
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

function Base.size(g::CuSparseMatrix, d::Integer)
    if 1 <= d <= 2
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

function Base.size(g::CuSparseArrayCSR{Tv,Ti,N}, d::Integer) where {Tv,Ti,N}
    if 1 <= d <= N
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

## sparse array interface

SparseArrays.nnz(g::AbstractCuSparseArray) = g.nnz
SparseArrays.nonzeros(g::AbstractCuSparseArray) = g.nzVal

SparseArrays.nonzeroinds(g::AbstractCuSparseVector) = g.iPtr

SparseArrays.rowvals(g::CuSparseMatrixCSC) = g.rowVal
SparseArrays.getcolptr(S::CuSparseMatrixCSC) = S.colPtr

function SparseArrays.findnz(S::MT) where {MT <: AbstractCuSparseMatrix}
    S2 = CuSparseMatrixCOO(S)
    I = S2.rowInd
    J = S2.colInd
    V = S2.nzVal

    # To make it compatible with the SparseArrays.jl version
    idxs = sortperm(J)
    I = I[idxs]
    J = J[idxs]
    V = V[idxs]

    return (I, J, V)
end

function SparseArrays.sparsevec(I::CuArray{Ti}, V::CuArray{Tv}, n::Integer) where {Ti,Tv}
    CuSparseVector(I, V, n)
end

LinearAlgebra.issymmetric(M::Union{CuSparseMatrixCSC,CuSparseMatrixCSR}) = size(M, 1) == size(M, 2) ? norm(M - transpose(M), Inf) == 0 : false
LinearAlgebra.ishermitian(M::Union{CuSparseMatrixCSC,CuSparseMatrixCSR}) = size(M, 1) == size(M, 2) ? norm(M - adjoint(M), Inf) == 0 : false
LinearAlgebra.issymmetric(M::Symmetric{CuSparseMatrixCSC}) = true
LinearAlgebra.ishermitian(M::Hermitian{CuSparseMatrixCSC}) = true

LinearAlgebra.istriu(M::UpperTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = true
LinearAlgebra.istril(M::UpperTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = false
LinearAlgebra.istriu(M::LowerTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = false
LinearAlgebra.istril(M::LowerTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = true

Hermitian{T}(Mat::CuSparseMatrix{T}) where T = Hermitian{T,typeof(Mat)}(Mat,'U')

SparseArrays.nnz(g::CuSparseMatrixBSR) = g.nnzb * g.blockDim * g.blockDim


## indexing

# translations
Base.getindex(A::AbstractCuSparseVector, ::Colon)          = copy(A)
Base.getindex(A::AbstractCuSparseMatrix, ::Colon, ::Colon) = copy(A)
Base.getindex(A::AbstractCuSparseMatrix, i, ::Colon)       = getindex(A, i, 1:size(A, 2))
Base.getindex(A::AbstractCuSparseMatrix, ::Colon, i)       = getindex(A, 1:size(A, 1), i)
Base.getindex(A::AbstractCuSparseMatrix, I::Tuple{Integer,Integer}) = getindex(A, I[1], I[2])

# column slices
function Base.getindex(x::CuSparseMatrixCSC, ::Colon, j::Integer)
    checkbounds(x, :, j)
    r1 = convert(Int, x.colPtr[j])
    r2 = convert(Int, x.colPtr[j+1]) - 1
    CuSparseVector(rowvals(x)[r1:r2], nonzeros(x)[r1:r2], size(x, 1))
end

function Base.getindex(x::CuSparseMatrixCSR, i::Integer, ::Colon)
    checkbounds(x, i, :)
    c1 = convert(Int, x.rowPtr[i])
    c2 = convert(Int, x.rowPtr[i+1]) - 1
    CuSparseVector(x.colVal[c1:c2], nonzeros(x)[c1:c2], size(x, 2))
end

# row slices
Base.getindex(A::CuSparseMatrixCSC, i::Integer, ::Colon) = CuSparseVector(sparse(A[i, 1:end]))  # TODO: optimize
Base.getindex(A::CuSparseMatrixCSR, ::Colon, j::Integer) = CuSparseVector(sparse(A[1:end, j]))  # TODO: optimize

function Base.getindex(A::CuSparseVector{Tv, Ti}, i::Integer) where {Tv, Ti}
    @boundscheck checkbounds(A, i)
    ii = searchsortedfirst(A.iPtr, convert(Ti, i))
    (ii > nnz(A) || A.iPtr[ii] != i) && return zero(Tv)
    A.nzVal[ii]
end

function Base.getindex(A::CuSparseMatrixCSC{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    r1 = Int(A.colPtr[i1])
    r2 = Int(A.colPtr[i1+1]-1)
    (r1 > r2) && return zero(T)
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    (r1 > r2 || rowvals(A)[r1] != i0) && return zero(T)
    nonzeros(A)[r1]
end

function Base.getindex(A::CuSparseMatrixCSR{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    c1 = Int(A.rowPtr[i0])
    c2 = Int(A.rowPtr[i0+1]-1)
    (c1 > c2) && return zero(T)
    c1 = searchsortedfirst(A.colVal, i1, c1, c2, Base.Order.Forward)
    (c1 > c2 || A.colVal[c1] != i1) && return zero(T)
    nonzeros(A)[c1]
end

function Base.getindex(A::CuSparseMatrixCOO{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    r1 = searchsortedfirst(A.rowInd, i0, Base.Order.Forward)
    (r1 > length(A.rowInd) || A.rowInd[r1] > i0) && return zero(T)
    r2 = searchsortedfirst(A.rowInd, i0+1, Base.Order.Forward)
    c1 = searchsortedfirst(A.colInd, i1, r1, r2, Base.Order.Forward)
    (c1 > r2 || A.colInd[c1] > i1) && return zero(T)
    nonzeros(A)[c1]
end

function Base.getindex(A::CuSparseMatrixBSR{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    i0_block, i0_idx = fldmod1(i0, A.blockDim)
    i1_block, i1_idx = fldmod1(i1, A.blockDim)
    block_idx = (i0_idx - 1) * A.blockDim + i1_idx - 1
    c1 = Int(A.rowPtr[i0_block])
    c2 = Int(A.rowPtr[i0_block+1]-1)
    (c1 > c2) && return zero(T)
    c1 = searchsortedfirst(A.colVal, i1_block, c1, c2, Base.Order.Forward)
    (c1 > c2 || A.colVal[c1] != i1_block) && return zero(T)
    nonzeros(A)[c1+block_idx]
end

# matrix slices
function Base.getindex(A::CuSparseArrayCSR{Tv, Ti, N}, ::Colon, ::Colon, idxs::Integer...) where {Tv, Ti, N}
    @boundscheck checkbounds(A, :, :, idxs...)
    CuSparseMatrixCSR(A.rowPtr[:,idxs...], A.colVal[:,idxs...], nonzeros(A)[:,idxs...], size(A)[1:2])
end

function Base.getindex(A::CuSparseArrayCSR{Tv, Ti, N}, i0::Integer, i1::Integer, idxs::Integer...) where {Tv, Ti, N}
    @boundscheck checkbounds(A, i0, i1, idxs...)
    CuSparseMatrixCSR(A.rowPtr[:,idxs...], A.colVal[:,idxs...], nonzeros(A)[:,idxs...], size(A)[1:2])[i0, i1]
end

## interop with sparse CPU arrays

# cpu to gpu
# NOTE: we eagerly convert the indices to Cint here to avoid additional conversion later on
CuSparseVector{T}(Vec::SparseVector) where {T} =
    CuSparseVector(CuVector{Cint}(Vec.nzind), CuVector{T}(Vec.nzval), length(Vec))
CuSparseVector{T}(Mat::SparseMatrixCSC) where {T} =
    size(Mat,2) == 1 ?
        CuSparseVector(CuVector{Cint}(Mat.rowval), CuVector{T}(Mat.nzval), size(Mat)[1]) :
        throw(ArgumentError("The input argument must have a single column"))
CuSparseMatrixCSC{T}(Vec::SparseVector) where {T} =
    CuSparseMatrixCSC{T}(CuVector{Cint}([1]), CuVector{Cint}(Vec.nzind),
                         CuVector{T}(Vec.nzval), size(Vec))
CuSparseMatrixCSC{T}(Mat::SparseMatrixCSC) where {T} =
    CuSparseMatrixCSC{T}(CuVector{Cint}(Mat.colptr), CuVector{Cint}(Mat.rowval),
                         CuVector{T}(Mat.nzval), size(Mat))
CuSparseMatrixCSR{T}(Mat::Transpose{Tv, <:SparseMatrixCSC}) where {T, Tv} =
    CuSparseMatrixCSR{T}(CuVector{Cint}(parent(Mat).colptr), CuVector{Cint}(parent(Mat).rowval),
                         CuVector{T}(parent(Mat).nzval), size(Mat))
CuSparseMatrixCSR{T}(Mat::Adjoint{Tv, <:SparseMatrixCSC}) where {T, Tv} =
    CuSparseMatrixCSR{T}(CuVector{Cint}(parent(Mat).colptr), CuVector{Cint}(parent(Mat).rowval),
                         CuVector{T}(conj.(parent(Mat).nzval)), size(Mat))
CuSparseMatrixCSC{T}(Mat::Union{Transpose{Tv, <:SparseMatrixCSC}, Adjoint{Tv, <:SparseMatrixCSC}}) where {T, Tv} = CuSparseMatrixCSC(CuSparseMatrixCSR{T}(Mat))
CuSparseMatrixCSR{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCSR(CuSparseMatrixCSC{T}(Mat))
CuSparseMatrixBSR{T}(Mat::SparseMatrixCSC, blockdim) where {T} = CuSparseMatrixBSR(CuSparseMatrixCSR{T}(Mat), blockdim)
CuSparseMatrixCOO{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCOO(CuSparseMatrixCSR{T}(Mat))

# untyped variants
CuSparseVector(x::AbstractSparseArray{T}) where {T} = CuSparseVector{T}(x)
CuSparseMatrixCSC(x::AbstractSparseArray{T}) where {T} = CuSparseMatrixCSC{T}(x)
CuSparseMatrixCSR(x::AbstractSparseArray{T}) where {T} = CuSparseMatrixCSR{T}(x)
CuSparseMatrixBSR(x::AbstractSparseArray{T}, blockdim) where {T} = CuSparseMatrixBSR{T}(x, blockdim)
CuSparseMatrixCOO(x::AbstractSparseArray{T}) where {T} = CuSparseMatrixCOO{T}(x)

# adjoint / transpose
CuSparseMatrixCSR(x::Transpose{T}) where {T} = CuSparseMatrixCSR{T}(x)
CuSparseMatrixCSR(x::Adjoint{T}) where {T} = CuSparseMatrixCSR{T}(x)
CuSparseMatrixCSC(x::Transpose{T}) where {T} = CuSparseMatrixCSC{T}(x)
CuSparseMatrixCSC(x::Adjoint{T}) where {T} = CuSparseMatrixCSC{T}(x)
CuSparseMatrixCOO(x::Transpose{T}) where {T} = CuSparseMatrixCOO{T}(x)
CuSparseMatrixCOO(x::Adjoint{T}) where {T} = CuSparseMatrixCOO{T}(x)

CuSparseMatrixCSR(x::Transpose{T,<:Union{CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO}}) where {T} = CuSparseMatrixCSR(_sptranspose(parent(x)))
CuSparseMatrixCSC(x::Transpose{T,<:Union{CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO}}) where {T} = CuSparseMatrixCSC(_sptranspose(parent(x)))
CuSparseMatrixCOO(x::Transpose{T,<:Union{CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO}}) where {T} = CuSparseMatrixCOO(_sptranspose(parent(x)))
CuSparseMatrixCSR(x::Adjoint{T,<:Union{CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO}}) where {T} = CuSparseMatrixCSR(_spadjoint(parent(x)))
CuSparseMatrixCSC(x::Adjoint{T,<:Union{CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO}}) where {T} = CuSparseMatrixCSC(_spadjoint(parent(x)))
CuSparseMatrixCOO(x::Adjoint{T,<:Union{CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO}}) where {T} = CuSparseMatrixCOO(_spadjoint(parent(x)))

# gpu to cpu
SparseVector(x::CuSparseVector) = SparseVector(length(x), Array(nonzeroinds(x)), Array(nonzeros(x)))
SparseMatrixCSC(x::CuSparseMatrixCSC) = SparseMatrixCSC(size(x)..., Array(x.colPtr), Array(rowvals(x)), Array(nonzeros(x)))
SparseMatrixCSC(x::CuSparseMatrixCSR) = SparseMatrixCSC(CuSparseMatrixCSC(x))  # no direct conversion (gpu_CSR -> gpu_CSC -> cpu_CSC)
SparseMatrixCSC(x::CuSparseMatrixBSR) = SparseMatrixCSC(CuSparseMatrixCSR(x))  # no direct conversion (gpu_BSR -> gpu_CSR -> gpu_CSC -> cpu_CSC)
SparseMatrixCSC(x::CuSparseMatrixCOO) = SparseMatrixCSC(CuSparseMatrixCSC(x))  # no direct conversion (gpu_COO -> gpu_CSC -> cpu_CSC)

# collect to Array
Base.collect(x::CuSparseVector) = collect(SparseVector(x))
Base.collect(x::CuSparseMatrixCSC) = collect(SparseMatrixCSC(x))
Base.collect(x::CuSparseMatrixCSR) = collect(SparseMatrixCSC(x))
Base.collect(x::CuSparseMatrixBSR) = collect(SparseMatrixCSC(x))
Base.collect(x::CuSparseMatrixCOO) = collect(SparseMatrixCSC(x))

Adapt.adapt_storage(::Type{CuArray}, xs::SparseVector) = CuSparseVector(xs)
Adapt.adapt_storage(::Type{CuArray}, xs::SparseMatrixCSC) = CuSparseMatrixCSC(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseVector) where {T} = CuSparseVector{T}(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseMatrixCSC) where {T} = CuSparseMatrixCSC{T}(xs)

Adapt.adapt_storage(::CUDA.CuArrayKernelAdaptor, xs::AbstractSparseArray) =
  adapt(CuArray, xs)
Adapt.adapt_storage(::CUDA.CuArrayKernelAdaptor, xs::AbstractSparseArray{<:AbstractFloat}) =
  adapt(CuArray{Float32}, xs)

Adapt.adapt_storage(::Type{Array}, xs::CuSparseVector) = SparseVector(xs)
Adapt.adapt_storage(::Type{Array}, xs::CuSparseMatrixCSC) = SparseMatrixCSC(xs)


## copying between sparse GPU arrays

function Base.copyto!(dst::CuSparseVector, src::CuSparseVector)
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent Sparse Vector size"))
    end
    resize!(nonzeroinds(dst), length(nonzeroinds(src)))
    resize!(nonzeros(dst), length(nonzeros(src)))
    copyto!(nonzeroinds(dst), nonzeroinds(src))
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = src.nnz
    dst
end

function Base.copyto!(dst::CuSparseMatrixCSC, src::CuSparseMatrixCSC)
    if size(dst) != size(src)
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resize!(dst.colPtr, length(src.colPtr))
    resize!(rowvals(dst), length(rowvals(src)))
    resize!(nonzeros(dst), length(nonzeros(src)))
    copyto!(dst.colPtr, src.colPtr)
    copyto!(rowvals(dst), rowvals(src))
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = src.nnz
    dst
end

function Base.copyto!(dst::CuSparseMatrixCSR, src::CuSparseMatrixCSR)
    if size(dst) != size(src)
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resize!(dst.rowPtr, length(src.rowPtr))
    resize!(dst.colVal, length(src.colVal))
    resize!(nonzeros(dst), length(nonzeros(src)))
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = src.nnz
    dst
end

function Base.copyto!(dst::CuSparseMatrixBSR, src::CuSparseMatrixBSR)
    if size(dst) != size(src)
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resize!(dst.rowPtr, length(src.rowPtr))
    resize!(dst.colVal, length(src.colVal))
    resize!(nonzeros(dst), length(nonzeros(src)))
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(nonzeros(dst), nonzeros(src))
    dst.dir = src.dir
    dst.nnzb = src.nnzb
    dst
end

function Base.copyto!(dst::CuSparseMatrixCOO, src::CuSparseMatrixCOO)
    if size(dst) != size(src)
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resize!(dst.rowInd, length(src.rowInd))
    resize!(dst.colInd, length(src.colInd))
    resize!(nonzeros(dst), length(nonzeros(src)))
    copyto!(dst.rowInd, src.rowInd)
    copyto!(dst.colInd, src.colInd)
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = src.nnz
    dst
end

Base.copy(Vec::CuSparseVector) = copyto!(similar(Vec), Vec)
Base.copy(Mat::CuSparseMatrixCSC) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseMatrixCSR) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseMatrixBSR) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseMatrixCOO) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseArrayCSR) = CuSparseArrayCSR(copy(Mat.rowPtr), copy(Mat.colVal), copy(nonzeros(Mat)), size(Mat))

# input/output

for (gpu, cpu) in [:CuSparseVector => :SparseVector]
    @eval function Base.show(io::IO, ::MIME"text/plain", x::$gpu)
        xnnz = length(nonzeros(x))
        print(io, length(x), "-element ", typeof(x), " with ", xnnz,
            " stored ", xnnz == 1 ? "entry" : "entries")
        if xnnz != 0
            println(io, ":")
            show(IOContext(io, :typeinfo => eltype(x)), $cpu(x))
        end
    end
end

for (gpu, cpu) in [:CuSparseMatrixCSC => :SparseMatrixCSC,
                   :CuSparseMatrixCSR => :SparseMatrixCSC,
                   :CuSparseMatrixBSR => :SparseMatrixCSC,
                   :CuSparseMatrixCOO => :SparseMatrixCSC]
    @eval Base.show(io::IOContext, x::$gpu) =
        show(io, $cpu(x))

    @eval function Base.show(io::IO, mime::MIME"text/plain", S::$gpu)
        xnnz = nnz(S)
        m, n = size(S)
        print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ",
                  xnnz == 1 ? "entry" : "entries")
        if !(m == 0 || n == 0)
            println(io, ":")
            io = IOContext(io, :typeinfo => eltype(S))
            if ndims(S) == 1
                show(io, $cpu(S))
            else
                # so that we get the nice Braille pattern
                Base.print_array(io, $cpu(S))
            end
        end
    end
end

function Base.show(io::IOContext, ::MIME"text/plain", A::CuSparseArrayCSR)
    xnnz = nnz(A)
    dims = join(size(A), "×")

    print(io, dims..., " ", typeof(A), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")

    if all(size(A) .> 0)
        println(io, ":")
        io = IOContext(io, :typeinfo => eltype(A))
        for (k, c) in enumerate(CartesianIndices(size(A)[3:end]))
            k > 1 && println(io, "\n")
            dims = join(c.I, ", ")
            println(io, "[:, :, $dims] =")
            Base.print_array(io, SparseMatrixCSC(A[:,:,c.I...]))
        end
    end
end


# interop with device arrays

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, x::CuSparseVector)
    return CuSparseDeviceVector(
        adapt(to, x.iPtr),
        adapt(to, x.nzVal),
        length(x), x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, x::CuSparseMatrixCSR)
    return CuSparseDeviceMatrixCSR(
        adapt(to, x.rowPtr),
        adapt(to, x.colVal),
        adapt(to, x.nzVal),
        size(x), x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, x::CuSparseMatrixCSC)
    return CuSparseDeviceMatrixCSC(
        adapt(to, x.colPtr),
        adapt(to, x.rowVal),
        adapt(to, x.nzVal),
        size(x), x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, x::CuSparseMatrixBSR)
    return CuSparseDeviceMatrixBSR(
        adapt(to, x.rowPtr),
        adapt(to, x.colVal),
        adapt(to, x.nzVal),
        size(x), x.blockDim,
        x.dir, x.nnzb
    )
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, x::CuSparseMatrixCOO)
    return CuSparseDeviceMatrixCOO(
        adapt(to, x.rowInd),
        adapt(to, x.colInd),
        adapt(to, x.nzVal),
        size(x), x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, x::CuSparseArrayCSR)
    return CuSparseDeviceArrayCSR(
        adapt(to, x.rowPtr),
        adapt(to, x.colVal),
        adapt(to, x.nzVal),
        size(x), x.nnz
    )
end

