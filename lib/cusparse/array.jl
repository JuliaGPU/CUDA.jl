# custom extension of CuArray in CUDArt for sparse vectors/matrices
# using CSC format for interop with Julia's native sparse functionality

export CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR, CuSparseMatrixCOO,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseVector

using LinearAlgebra: BlasFloat
using SparseArrays: nonzeroinds, dimlub, getcolptr, rowvals

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
                                   nzVal::CuVector, dims::NTuple{2,Int}) where {Tv, Ti<:Integer}
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
    nnz::Ti

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
                                   nzVal::CuVector, dims::NTuple{2,Int}=(dimlub(rowInd),dimlub(colInd)),
                                   nnz::Integer=length(nzVal)) where {Tv, Ti}
        new{Tv, Ti}(rowInd,colInd,nzVal,dims,nnz)
    end
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

## similar for CSC,CSR with dims, eltype arguments, adapted from SparseArrays.jl. NOTE: calling similar() on a wrapped CuSparseMatrixBSR/COO will result in a CuArray.
# parent method for similar that preserves stored-entry structure (for when new and old dims match)
function _cusparsesimilar(S::CuSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    newcolptr = copyto!(similar(getcolptr(S), TiNew), getcolptr(S))
    newrowval = copyto!(similar(rowvals(S), TiNew), rowvals(S))
    return CuSparseMatrixCSC(newcolptr, newrowval, similar(nonzeros(S), TvNew), size(S))
end
function _cusparsesimilar(S::CuSparseMatrixCSR, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    newrowptr = copyto!(similar(getrowptr(S), TiNew), getrowptr(S))
    newcolval = copyto!(similar(colvals(S), TiNew), colvals(S))
    return CuSparseMatrixCSR(newrowptr, newcolval, similar(nonzeros(S), TvNew), size(S))
end
# parent methods for similar that preserves only storage space (for when new and old dims differ)
_cusparsesimilar(S::CuSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, dims::Dims{2}) where {TvNew,TiNew} =
    CuSparseMatrixCSC(CUDA.fill(one(TiNew), last(dims)+1), similar(rowvals(S), TiNew), similar(nonzeros(S), TvNew),dims)
_cusparsesimilar(S::CuSparseMatrixCSR, ::Type{TvNew}, ::Type{TiNew}, dims::Dims{2}) where {TvNew,TiNew} =
    CuSparseMatrixCSR(CUDA.fill(one(TiNew), first(dims)+1), similar(colvals(S), TiNew), similar(nonzeros(S), TvNew),dims)
# parent method for similar that allocates an empty sparse vector (when new dims are single)
_cusparsesimilar(S::CuSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, dims::Dims{1}) where {TvNew,TiNew} =
    CuSparseVector(similar(rowvals(S), TiNew, 0), similar(nonzeros(S), TvNew, 0),dims...)
_cusparsesimilar(S::CuSparseMatrixCSR, ::Type{TvNew}, ::Type{TiNew}, dims::Dims{1}) where {TvNew,TiNew} =
    CuSparseVector(similar(colvals(S), TiNew, 0), similar(nonzeros(S), TvNew, 0),dims...)

"""
Utility union type of [`CuSparseMatrixCSC`](@ref), [`CuSparseMatrixCSR`](@ref), for which similar is implemented
"""
const CuSparseMatrixCSCR{Tv, Ti} = Union{
    CuSparseMatrixCSC{Tv, Ti},
    CuSparseMatrixCSR{Tv, Ti}
}
# The following methods hook into the AbstractArray similar hierarchy. The first method
# covers similar(A[, Tv]) calls, which preserve stored-entry structure, and the latter
# methods cover similar(A[, Tv], shape...) calls, which preserve storage space when the shape
# calls for a two-dimensional result.
Base.similar(S::CuSparseMatrixCSCR{<:Any,Ti}, ::Type{TvNew}) where {Ti,TvNew} = _cusparsesimilar(S, TvNew, Ti)
Base.similar(S::CuSparseMatrixCSCR{<:Any,Ti}, ::Type{TvNew}, dims::Union{Dims{1},Dims{2}}) where {Ti,TvNew} =
    _cusparsesimilar(S, TvNew, Ti, dims)
# The following methods cover similar(A, Tv, Ti[, shape...]) calls, which specify the
# result's index type in addition to its entry type, and aren't covered by the hooks above.
# The calls without shape again preserve stored-entry structure, whereas those with shape
# preserve storage space when the shape calls for a two-dimensional result.
Base.similar(S::CuSparseMatrixCSCR, ::Type{TvNew}, ::Type{TiNew}) where{TvNew,TiNew} =
    _cusparsesimilar(S, TvNew, TiNew)
Base.similar(S::CuSparseMatrixCSCR, ::Type{TvNew}, ::Type{TiNew}, dims::Union{Dims{1},Dims{2}}) where {TvNew,TiNew} =
    _cusparsesimilar(S, TvNew, TiNew, dims)
Base.similar(S::CuSparseMatrixCSCR, ::Type{TvNew}, ::Type{TiNew}, m::Integer) where {TvNew,TiNew} =
    _cusparsesimilar(S, TvNew, TiNew, (m,))
Base.similar(S::CuSparseMatrixCSCR, ::Type{TvNew}, ::Type{TiNew}, m::Integer, n::Integer) where {TvNew,TiNew} =
    _cusparsesimilar(S, TvNew, TiNew, (m, n))

# Handles cases where CSC/CSR are reshaped to more than two dimensions, and all cases above for COO/BSR.
Base.similar(a::AbstractCuSparseArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =
CuArray{T,N}(undef, dims)               

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
                                   
SparseArrays.getcolptr(g::CuSparseMatrixCSC) = g.colPtr
SparseArrays.rowvals(g::CuSparseMatrixCSC) = g.rowVal
getrowptr(g::CuSparseMatrixCSR) = g.rowPtr
colvals(g::CuSparseMatrixCSR) = g.colVal


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
CuSparseMatrixCSR{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCSR(CuSparseMatrixCSC{T}(Mat))
CuSparseMatrixBSR{T}(Mat::SparseMatrixCSC, blockdim) where {T} = CuSparseMatrixBSR(CuSparseMatrixCSR{T}(Mat), blockdim)
CuSparseMatrixCOO{T}(Mat::SparseMatrixCSC) where {T} = CuSparseMatrixCOO(CuSparseMatrixCSR{T}(Mat))

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
