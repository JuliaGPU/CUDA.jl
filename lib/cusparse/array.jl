# custom extension of CuArray in CUDArt for sparse vectors/matrices
# using CSC format for interop with Julia's native sparse functionality

export CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR, CuSparseMatrixCOO,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseVector

using LinearAlgebra: BlasFloat
using SparseArrays: nonzeroinds, dimlub

abstract type AbstractCuSparseArray{Tv, Ti, N} <: AbstractSparseArray{Tv, Ti, N} end
const AbstractCuSparseVector{Tv, Ti} = AbstractCuSparseArray{Tv, Ti, 1}
const AbstractCuSparseMatrix{Tv, Ti} = AbstractCuSparseArray{Tv, Ti, 2}

Base.convert(T::Type{<:AbstractCuSparseArray}, m::AbstractArray) = m isa T ? m : T(m)

mutable struct CuSparseVector{Tv, Ti} <: AbstractCuSparseVector{Tv, Ti}
    iPtr::CuVector{Ti}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Ti

    function CuSparseVector{Tv, Ti}(iPtr::CuVector{<:Integer}, nzVal::CuVector,
                                dims::Integer) where {Tv, Ti <: Integer}
        new{Tv, Ti}(iPtr, nzVal, (dims,1), length(nzVal))
    end
end

CuSparseVector{Tv, Ti}(Vec::CuSparseVector{Tv, Ti}) where {Tv, Ti} = Vec
CuSparseVector(Vec::CuSparseVector{Tv, Ti}) where {Tv, Ti} = CuSparseVector{Tv, Ti}(Vec)

function CuSparseVector{Tv, Ti}(Vec::CuSparseVector) where {Tv, Ti}
    CuSparseVector{Tv, Ti}(CuVector{Ti}(Vec.iPtr), CuVector{Tv}(Vec.nzVal), Vec.dims)
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

CuSparseMatrixCSC{Tv, Ti}(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = A
function CuSparseMatrixCSC{Tv, Ti}(A::CuSparseMatrixCSC) where {Tv, Ti}
    CuSparseMatrixCSC{Tv, Ti}(CuVector{Ti}(A.colPtr), CuVector{Ti}(A.rowVal), CuVector{Tv}(A.nzVal), A.dims)
end

CuSparseMatrixCSC(A::AbstractCuSparseMatrix{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixCSC{Tv, Ti}(A)

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
                                   nzVal::CuVector, dims::NTuple{2,Int}) where {Tv, Ti<:Integer}
        new{Tv, Ti}(rowPtr, colVal, nzVal, dims, length(nzVal))
    end
end

CuSparseMatrixCSR{Tv, Ti}(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti} = A
function CuSparseMatrixCSR{Tv, Ti}(A::CuSparseMatrixCSR) where {Tv, Ti}
    CuSparseMatrixCSR{Tv, Ti}(CuVector{Ti}(A.rowPtr), CuVector{Ti}(A.colVal), CuVector{Tv}(A.nzVal), A.dims)
end

CuSparseMatrixCSR(A::AbstractCuSparseMatrix{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixCSR{Tv, Ti}(A)

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
    nnz::Ti

    function CuSparseMatrixBSR{Tv, Ti}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer},
                                   blockDim::Integer, dir::SparseChar, nnz::Integer) where {Tv, Ti<:Integer}
        new{Tv, Ti}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)
    end
end

CuSparseMatrixBSR{Tv, Ti}(A::CuSparseMatrixBSR{Tv, Ti}) where {Tv, Ti} = A
function CuSparseMatrixBSR{Tv, Ti}(A::CuSparseMatrixBSR) where {Tv, Ti}
    return CuSparseMatrixBSR{Tv, Ti}(
        CuVector{Ti}(A.rowPtr),
        CuVector{Ti}(A.colVal),
        CuVector{Tv}(A.nzVal),
        A.dims, Ti(A.blockDim),
        A.dir, A.nnz
    )
end

CuSparseMatrixBSR(A::AbstractCuSparseMatrix{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixBSR{Tv, Ti}(A)

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
                                   nzVal::CuVector, dims::NTuple{2,Int}=(dimlub(rowInd),dimlub(colInd)),
                                   nnz::Integer=length(nzVal)) where {Tv, Ti}
        new{Tv, Ti}(rowInd,colInd,nzVal,dims,nnz)
    end
end

function CuSparseMatrixCOO{Tv, Ti}(A::CuSparseMatrixCOO) where {Tv, Ti}
    return CuSparseMatrixCOO{Tv, Ti}(
        CuVector{Ti}(A.rowInd),
        CuVector{Ti}(A.colInd),
        CuVector{Tv}(A.nzVal),
        A.dims, A.nnz
    )
end

CuSparseMatrixCOO(A::CuSparseMatrixCOO) = A

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


# NOTE: we use Cint as default Ti on CUDA instead of Int to provide
# maximum compatiblity to old CUSPARSE APIs
function CuSparseVector{Tv}(iPtr::CuVector{<:Integer}, nzVal::CuVector, dims::Integer) where {Tv}
    CuSparseVector{Tv, Cint}(convert(CuVector{Cint}, iPtr), nzVal, dims)
end

function CuSparseMatrixCSC{Tv}(colPtr::CuVector{<:Integer}, rowVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer}) where {Tv}
    CuSparseMatrixCSC{Tv, Cint}(colPtr, rowVal, nzVal, dims)
end

function CuSparseMatrixCSR{Tv}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,Int}) where {Tv}
    CuSparseMatrixCSR{Tv, Cint}(rowPtr, colVal, nzVal, dims)
end

function CuSparseMatrixBSR{Tv}(rowPtr::CuVector{<:Integer}, colVal::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,<:Integer},
                                   blockDim::Integer, dir::SparseChar, nnz::Integer) where {Tv}
    CuSparseMatrixBSR{Tv, Cint}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)
end

function CuSparseMatrixCOO{Tv}(rowInd::CuVector{<:Integer}, colInd::CuVector{<:Integer},
                                   nzVal::CuVector, dims::NTuple{2,Int}=(dimlub(rowInd),dimlub(colInd)),
                                   nnz::Integer=length(nzVal)) where {Tv}
    CuSparseMatrixCOO{Tv, Cint}(rowInd,colInd,nzVal,dims,nnz)
end

## convenience constructors
CuSparseVector(iPtr::DenseCuArray{<:Integer}, nzVal::DenseCuArray{T}, dims::Int) where {T} =
    CuSparseVector{T}(iPtr, nzVal, dims)

CuSparseMatrixCSC(colPtr::DenseCuArray{<:Integer}, rowVal::DenseCuArray{<:Integer},
                  nzVal::DenseCuArray{T}, dims::NTuple{2,Int}) where {T} =
    CuSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims)

CuSparseMatrixCSR(rowPtr::DenseCuArray, colVal::DenseCuArray, nzVal::DenseCuArray{T}, dims::NTuple{2,Int}) where T =
    CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims)

CuSparseMatrixBSR(rowPtr::DenseCuArray, colVal::DenseCuArray, nzVal::DenseCuArray{T}, blockDim, dir, nnz,
                  dims::NTuple{2,Int}) where T =
    CuSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)

Base.similar(Vec::CuSparseVector) = CuSparseVector(copy(nonzeroinds(Vec)), similar(nonzeros(Vec)), Vec.dims[1])
Base.similar(Mat::CuSparseMatrixCSC) = CuSparseMatrixCSC(copy(Mat.colPtr), copy(rowvals(Mat)), similar(nonzeros(Mat)), Mat.dims)
Base.similar(Mat::CuSparseMatrixCSR) = CuSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), Mat.dims)
Base.similar(Mat::CuSparseMatrixBSR) = CuSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), Mat.blockDim, Mat.dir, nnz(Mat), Mat.dims)


## array interface

Base.length(g::CuSparseVector) = prod(g.dims)
Base.size(g::CuSparseVector) = g.dims
Base.ndims(g::CuSparseVector) = 1

Base.length(g::CuSparseMatrix) = prod(g.dims)
Base.size(g::CuSparseMatrix) = g.dims
Base.ndims(g::CuSparseMatrix) = 2

function Base.size(g::CuSparseVector, d::Integer)
    if d == 1
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

function Base.size(g::CuSparseMatrix, d::Integer)
    if d in [1, 2]
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

Base.eltype(g::CuSparseMatrix{T}) where T = T


## sparse array interface

SparseArrays.nnz(g::AbstractCuSparseArray) = g.nnz
SparseArrays.nonzeros(g::AbstractCuSparseArray) = g.nzVal

SparseArrays.nonzeroinds(g::AbstractCuSparseVector) = g.iPtr

SparseArrays.rowvals(g::CuSparseMatrixCSC) = g.rowVal

LinearAlgebra.issymmetric(M::Union{CuSparseMatrixCSC,CuSparseMatrixCSR}) = false
LinearAlgebra.ishermitian(M::Union{CuSparseMatrixCSC,CuSparseMatrixCSR}) = false
LinearAlgebra.issymmetric(M::Symmetric{CuSparseMatrixCSC}) = true
LinearAlgebra.ishermitian(M::Hermitian{CuSparseMatrixCSC}) = true

LinearAlgebra.istriu(M::UpperTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = true
LinearAlgebra.istril(M::UpperTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = false
LinearAlgebra.istriu(M::LowerTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = false
LinearAlgebra.istril(M::LowerTriangular{T,S}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix} = true

Hermitian{T}(Mat::CuSparseMatrix{T}) where T = Hermitian{T,typeof(Mat)}(Mat,'U')


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

function Base.getindex(A::CuSparseMatrixCSC{T}, i0::Integer, i1::Integer) where T
    m, n = size(A)
    if !(1 <= i0 <= m && 1 <= i1 <= n)
        throw(BoundsError())
    end
    r1 = Int(A.colPtr[i1])
    r2 = Int(A.colPtr[i1+1]-1)
    (r1 > r2) && return zero(T)
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? zero(T) : nonzeros(A)[r1]
end

function Base.getindex(A::CuSparseMatrixCSR{T}, i0::Integer, i1::Integer) where T
    m, n = size(A)
    if !(1 <= i0 <= m && 1 <= i1 <= n)
        throw(BoundsError())
    end
    c1 = Int(A.rowPtr[i0])
    c2 = Int(A.rowPtr[i0+1]-1)
    (c1 > c2) && return zero(T)
    c1 = searchsortedfirst(A.colVal, i1, c1, c2, Base.Order.Forward)
    ((c1 > c2) || (A.colVal[c1] != i1)) ? zero(T) : nonzeros(A)[c1]
end

function SparseArrays._spgetindex(m::Integer, nzind::CuVector{Ti}, nzval::CuVector{Tv},
                                  i::Integer) where {Tv,Ti}
    ii = searchsortedfirst(nzind, convert(Ti, i))
    (ii <= m && nzind[ii] == i) ? nzval[ii] : zero(Tv)
end

## interop with sparse CPU arrays

# cpu to gpu
CuSparseVector{Tv, Ti}(Vec::SparseVector) where {Tv, Ti} =
    CuSparseVector(CuVector{Ti}(Vec.nzind), CuVector{Tv}(Vec.nzval), length(Vec))
CuSparseVector{Tv, Ti}(Mat::SparseMatrixCSC) where {Tv, Ti} =
    size(Mat,2) == 1 ?
        CuSparseVector(CuVector{Ti}(Mat.rowval), CuVector{Tv}(Mat.nzval), size(Mat)[1]) :
        throw(ArgumentError("The input argument must have a single column"))
CuSparseMatrixCSC{Tv, Ti}(Vec::SparseVector) where {Tv, Ti} =
    CuSparseMatrixCSC{Tv, Ti}(CuVector{Ti}([1]), CuVector{Ti}(Vec.nzind),
                         CuVector{Tv}(Vec.nzval), size(Vec))
CuSparseMatrixCSC{Tv, Ti}(Mat::SparseMatrixCSC) where {Tv, Ti} =
    CuSparseMatrixCSC{Tv, Ti}(CuVector{Ti}(Mat.colptr), CuVector{Ti}(Mat.rowval),
                         CuVector{Tv}(Mat.nzval), size(Mat))
CuSparseMatrixCSR{Tv, Ti}(Mat::Transpose{T, <:SparseMatrixCSC}) where {T, Tv, Ti} =
    CuSparseMatrixCSR{Tv, Ti}(CuVector{Ti}(parent(Mat).colptr), CuVector{Ti}(parent(Mat).rowval),
                         CuVector{Tv}(parent(Mat).nzval), size(Mat))
CuSparseMatrixCSR{Tv, Ti}(Mat::Adjoint{T, <:SparseMatrixCSC}) where {T, Tv, Ti} =
    CuSparseMatrixCSR{Tv, Ti}(CuVector{Ti}(parent(Mat).colptr), CuVector{Ti}(parent(Mat).rowval),
                         CuVector{Tv}(conj.(parent(Mat).nzval)), size(Mat))

# NOTE: this requires generic format conversion
# CUDA doesn't support generic indices type
# for format, thus we only support Cint
CuSparseMatrixCSR{Tv, Cint}(Mat::SparseMatrixCSC) where {Tv} = CuSparseMatrixCSR(CuSparseMatrixCSC{Tv, Cint}(Mat))
CuSparseMatrixBSR{Tv, Cint}(Mat::SparseMatrixCSC, blockdim) where {Tv} = CuSparseMatrixBSR(CuSparseMatrixCSR{Tv, Cint}(Mat), blockdim)
CuSparseMatrixCOO{Tv, Cint}(Mat::SparseMatrixCSC) where {Tv} = CuSparseMatrixCOO(CuSparseMatrixCSR{Tv, Cint}(Mat))


# NOTE: we eagerly convert the indices to Cint here to avoid additional conversion later on
CuSparseVector{T}(Vec::SparseVector) where {T} = CuSparseVector{T, Cint}(Vec)
CuSparseVector{T}(Mat::SparseMatrixCSC) where {T} = CuSparseVector{T, Cint}(Mat)
CuSparseMatrixCSC{T}(Vec::SparseVector) where {T} = CuSparseMatrixCSC{T, Cint}(Vec)
CuSparseMatrixCSC{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCSC{T, Cint}(Mat)
CuSparseMatrixCSR{T}(Mat::Transpose{Tv, <:SparseMatrixCSC}) where {T, Tv} = CuSparseMatrixCSR{T, Cint}(Mat)
CuSparseMatrixCSR{T}(Mat::Adjoint{Tv, <:SparseMatrixCSC}) where {T, Tv} = CuSparseMatrixCSR{T, Cint}(Mat)
CuSparseMatrixCSR{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCSR{T, Cint}(Mat)
CuSparseMatrixBSR{T}(Mat::SparseMatrixCSC, blockdim) where {T} = CuSparseMatrixBSR{T, Cint}(Mat, blockdim)
CuSparseMatrixCOO{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCOO{T, Cint}(Mat)

# untyped variants
CuSparseVector(x::AbstractSparseArray{T}) where {T} = CuSparseVector{T}(x)
CuSparseMatrixCSC(x::AbstractSparseArray{T}) where {T} = CuSparseMatrixCSC{T}(x)
CuSparseMatrixCSR(x::AbstractSparseArray{T}) where {T} = CuSparseMatrixCSR{T}(x)
CuSparseMatrixBSR(x::AbstractSparseArray{T}, blockdim) where {T} = CuSparseMatrixBSR{T}(x, blockdim)
CuSparseMatrixCOO(x::AbstractSparseArray{T}) where {T} = CuSparseMatrixCOO{T}(x)
CuSparseMatrixCSR(x::Transpose{T}) where {T} = CuSparseMatrixCSR{T}(x)
CuSparseMatrixCSR(x::Adjoint{T}) where {T} = CuSparseMatrixCSR{T}(x)
CuSparseMatrixCSC(x::Transpose{T}) where {T} = CuSparseMatrixCSC{T}(x)
CuSparseMatrixCSC(x::Adjoint{T}) where {T} = CuSparseMatrixCSC{T}(x)

# gpu to cpu
SparseVector(x::CuSparseVector) = SparseVector(length(x), Array(nonzeroinds(x)), Array(nonzeros(x)))
SparseMatrixCSC(x::CuSparseMatrixCSC) = SparseMatrixCSC(size(x)..., Array(x.colPtr), Array(rowvals(x)), Array(nonzeros(x)))
SparseMatrixCSC(x::CuSparseMatrixCSR) = SparseMatrixCSC(CuSparseMatrixCSC(x))  # no direct conversion
SparseMatrixCSC(x::CuSparseMatrixBSR) = SparseMatrixCSC(CuSparseMatrixCSR(x))  # no direct conversion
SparseMatrixCSC(x::CuSparseMatrixCOO) = SparseMatrixCSC(CuSparseMatrixCSR(x))  # no direct conversion

# collect to Array
Base.collect(x::CuSparseVector) = collect(SparseVector(x))
Base.collect(x::CuSparseMatrixCSC) = collect(SparseMatrixCSC(x))
Base.collect(x::CuSparseMatrixCSR) = collect(SparseMatrixCSC(x))
Base.collect(x::CuSparseMatrixBSR) = collect(CuSparseMatrixCSR(x))  # no direct conversion
Base.collect(x::CuSparseMatrixCOO) = collect(CuSparseMatrixCSR(x))  # no direct conversion

Adapt.adapt_storage(::Type{CuArray}, xs::SparseVector) = CuSparseVector(xs)
Adapt.adapt_storage(::Type{CuArray}, xs::SparseMatrixCSC) = CuSparseMatrixCSC(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseVector) where {T} = CuSparseVector{T}(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseMatrixCSC) where {T} = CuSparseMatrixCSC{T}(xs)

Adapt.adapt_storage(::CUDA.CuArrayAdaptor, xs::AbstractSparseArray) =
  adapt(CuArray, xs)
Adapt.adapt_storage(::CUDA.CuArrayAdaptor, xs::AbstractSparseArray{<:AbstractFloat}) =
  adapt(CuArray{Float32}, xs)

Adapt.adapt_storage(::Type{Array}, xs::CuSparseVector) = SparseVector(xs)
Adapt.adapt_storage(::Type{Array}, xs::CuSparseMatrixCSC) = SparseMatrixCSC(xs)


## copying between sparse GPU arrays

function Base.copyto!(dst::CuSparseVector, src::CuSparseVector)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Vector size"))
    end
    copyto!(nonzeroinds(dst), nonzeroinds(src))
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = nnz(src)
    dst
end

function Base.copyto!(dst::CuSparseMatrixCSC, src::CuSparseMatrixCSC)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.colPtr, src.colPtr)
    copyto!(rowvals(dst), rowvals(src))
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = nnz(src)
    dst
end

function Base.copyto!(dst::CuSparseMatrixCSR, src::CuSparseMatrixCSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = nnz(src)
    dst
end

function Base.copyto!(dst::CuSparseMatrixBSR, src::CuSparseMatrixBSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(nonzeros(dst), nonzeros(src))
    dst.dir = src.dir
    dst.nnz = nnz(src)
    dst
end

function Base.copyto!(dst::CuSparseMatrixCOO, src::CuSparseMatrixCOO)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.rowInd, src.rowInd)
    copyto!(dst.colInd, src.colInd)
    copyto!(nonzeros(dst), nonzeros(src))
    dst.nnz = nnz(src)
    dst
end

Base.copy(Vec::CuSparseVector) = copyto!(similar(Vec), Vec)
Base.copy(Mat::CuSparseMatrixCSC) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseMatrixCSR) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseMatrixBSR) = copyto!(similar(Mat), Mat)
Base.copy(Mat::CuSparseMatrixCOO) = copyto!(similar(Mat), Mat)


# input/output

for (gpu, cpu) in [CuSparseVector => SparseVector]
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

for (gpu, cpu) in [CuSparseMatrixCSC => SparseMatrixCSC,
                   CuSparseMatrixCSR => SparseMatrixCSC,
                   CuSparseMatrixBSR => SparseMatrixCSC,
                   CuSparseMatrixCOO => SparseMatrixCSC]
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


# interop with device arrays

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseVector)
    return CuSparseDeviceVector(
        adapt(to, x.iPtr),
        adapt(to, x.nzVal),
        x.dims, x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseMatrixCSR)
    return CuSparseDeviceMatrixCSR(
        adapt(to, x.rowPtr),
        adapt(to, x.colVal),
        adapt(to, x.nzVal),
        x.dims, x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseMatrixCSC)
    return CuSparseDeviceMatrixCSC(
        adapt(to, x.colPtr),
        adapt(to, x.rowVal),
        adapt(to, x.nzVal),
        x.dims, x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseMatrixBSR)
    return CuSparseDeviceMatrixBSR(
        adapt(to, x.rowPtr),
        adapt(to, x.colVal),
        adapt(to, x.nzVal),
        x.dims, x.blockDim,
        x.dir, x.nnz
    )
end

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseMatrixCOO)
    return CuSparseDeviceMatrixCOO(
        adapt(to, x.rowInd),
        adapt(to, x.colInd),
        adapt(to, x.nzVal),
        x.dims, x.nnz
    )
end
