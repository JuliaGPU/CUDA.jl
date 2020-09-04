# custom extension of CuArray in CUDArt for sparse vectors/matrices
# using CSC format for interop with Julia's native sparse functionality

export CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseVector

using LinearAlgebra: BlasFloat

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

function CUDA.unsafe_free!(xs::CuSparseVector)
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
Utility union type of [`CuSparseMatrixCSC`](@ref), [`CuSparseMatrixCSR`](@ref), and
[`CuSparseMatrixBSR`](@ref).
"""
const CuSparseMatrix{T} = Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T}, CuSparseMatrixBSR{T}}


## convenience constructors

CuSparseVector(iPtr::Vector{Ti}, nzVal::Vector{T}, dims::Int) where {T<:BlasFloat, Ti<:Integer} =
    CuSparseVector{T}(CuArray(convert(Vector{Cint},iPtr)), CuArray(nzVal), dims, convert(Cint,length(nzVal)))
CuSparseVector(iPtr::CuArray{Ti}, nzVal::CuArray{T}, dims::Int) where {T<:BlasFloat, Ti<:Integer} =
    CuSparseVector{T}(iPtr, nzVal, dims, convert(Cint,length(nzVal)))

CuSparseMatrixCSC(colPtr::Vector{Ti}, rowVal::Vector{Ti}, nzVal::Vector{T}, dims::NTuple{2,Int}) where {T<:BlasFloat,Ti<:Integer} =
    CuSparseMatrixCSC{T}(CuArray(convert(Vector{Cint},colPtr)), CuArray(convert(Vector{Cint},rowVal)), CuArray(nzVal), dims, convert(Cint,length(nzVal)))
CuSparseMatrixCSC(colPtr::CuArray{Ti}, rowVal::CuArray{Ti}, nzVal::CuArray{T}, dims::NTuple{2,Int}) where {T<:BlasFloat,Ti<:Integer} =
    CuSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)))
CuSparseMatrixCSC(colPtr::CuArray{Ti}, rowVal::CuArray{Ti}, nzVal::CuArray{T}, nnz, dims::NTuple{2,Int}) where {T<:BlasFloat,Ti<:Integer} =
    CuSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, nnz)

CuSparseMatrixCSR(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, dims::NTuple{2,Int}) where T =
    CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)))
CuSparseMatrixCSR(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, nnz, dims::NTuple{2,Int}) where T =
    CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, nnz)

CuSparseMatrixBSR(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, blockDim, dir, nnz, dims::NTuple{2,Int}) where T =
    CuSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)

Base.similar(Vec::CuSparseVector) = CuSparseVector(copy(Vec.iPtr), similar(Vec.nzVal), Vec.dims[1])
Base.similar(Mat::CuSparseMatrixCSC) = CuSparseMatrixCSC(copy(Mat.colPtr), copy(Mat.rowVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
Base.similar(Mat::CuSparseMatrixCSR) = CuSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
Base.similar(Mat::CuSparseMatrixBSR) = CuSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.blockDim, Mat.dir, Mat.nnz, Mat.dims)


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
    CuSparseVector(x.rowVal[r1:r2], x.nzVal[r1:r2], size(x, 1))
end

function Base.getindex(x::CuSparseMatrixCSR, i::Integer, ::Colon)
    checkbounds(x, :, i)
    c1 = convert(Int, x.rowPtr[i])
    c2 = convert(Int, x.rowPtr[i+1]) - 1
    CuSparseVector(x.colVal[c1:c2], x.nzVal[c1:c2], size(x, 2))
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
    r1 = searchsortedfirst(A.rowVal, i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (A.rowVal[r1] != i0)) ? zero(T) : A.nzVal[r1]
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
    ((c1 > c2) || (A.colVal[c1] != i1)) ? zero(T) : A.nzVal[c1]
end

function SparseArrays._spgetindex(m::Integer, nzind::CuVector{Ti}, nzval::CuVector{Tv},
                                  i::Integer) where {Tv,Ti}
    ii = searchsortedfirst(nzind, convert(Ti, i))
    (ii <= m && nzind[ii] == i) ? nzval[ii] : zero(Tv)
end


## interop with sparse CPU arrays

# cpu to gpu
CuSparseVector(Vec::SparseVector) = CuSparseVector(Vec.nzind, Vec.nzval, length(Vec))
CuSparseVector(Mat::SparseMatrixCSC) = size(Mat,2) == 1 ? CuSparseVector(Mat.rowval, Mat.nzval, size(Mat)[1]) : throw(ArgumentError("The input argument must have a single column"))
CuSparseMatrixCSC(Vec::SparseVector) = CuSparseMatrixCSC([1], Vec.nzind, Vec.nzval, size(Vec))
CuSparseMatrixCSC(Mat::SparseMatrixCSC) = CuSparseMatrixCSC(Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))
CuSparseMatrixCSR(Mat::SparseMatrixCSC) = switch2csr(CuSparseMatrixCSC(Mat))
CuSparseMatrixBSR(Mat::SparseMatrixCSC; blockdim) = switch2bsr(CuSparseMatrixCSC(Mat), convert(Cint, blockdim))

# gpu to cpu
SparseVector(x::CuSparseVector) = SparseVector(length(x), Array(x.iPtr), Array(x.nzVal))
SparseMatrixCSC(x::CuSparseMatrixCSC) = SparseMatrixCSC(size(x)..., Array(x.colPtr), Array(x.rowVal), Array(x.nzVal))
function SparseMatrixCSC(Mat::CuSparseMatrixCSR)
    rowPtr = Array(Mat.rowPtr)
    colVal = Array(Mat.colVal)
    nzVal  = Array(Mat.nzVal)
    #construct Is
    I = similar(colVal)
    counter = 1
    for row = 1 : size(Mat)[1], k = rowPtr[row] : (rowPtr[row+1]-1)
        I[counter] = row
        counter += 1
    end
    return sparse(I,colVal,nzVal,Mat.dims[1],Mat.dims[2])
end

Base.collect(x::CuSparseVector) = SparseVector(x)
Base.collect(x::CuSparseMatrixCSC) = SparseMatrixCSC(x)
Base.collect(x::CuSparseMatrixCSR) = SparseMatrixCSC(x)


## interop between sparse GPU arrays

function Base.copyto!(dst::CuSparseVector, src::CuSparseVector)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Vector size"))
    end
    copyto!(dst.iPtr, src.iPtr)
    copyto!(dst.nzVal, src.nzVal)
    dst.nnz = src.nnz
    dst
end

function Base.copyto!(dst::CuSparseMatrixCSC, src::CuSparseMatrixCSC)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.colPtr, src.colPtr)
    copyto!(dst.rowVal, src.rowVal)
    copyto!(dst.nzVal, src.nzVal)
    dst.nnz = src.nnz
    dst
end

function Base.copyto!(dst::CuSparseMatrixCSR, src::CuSparseMatrixCSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(dst.nzVal, src.nzVal)
    dst.nnz = src.nnz
    dst
end

function Base.copyto!(dst::CuSparseMatrixBSR, src::CuSparseMatrixBSR)
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

Base.copy(Vec::CuSparseVector) = copyto!(similar(Vec),Vec)
Base.copy(Mat::CuSparseMatrixCSC) = copyto!(similar(Mat),Mat)
Base.copy(Mat::CuSparseMatrixCSR) = copyto!(similar(Mat),Mat)
Base.copy(Mat::CuSparseMatrixBSR) = copyto!(similar(Mat),Mat)


# input/output

Base.show(io::IOContext, x::CuSparseVector) =
    show(io, SparseVector(x))

Base.show(io::IOContext, x::CuSparseMatrixCSC) =
    show(io, SparseMatrixCSC(x))

Base.show(io::IO, S::CuSparseMatrixCSC) = Base.show(convert(IOContext, io), S)
function Base.show(io::IO, ::MIME"text/plain", S::CuSparseMatrixCSC)
    xnnz = nnz(S)
    m, n = size(S)
    print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if !(m == 0 || n == 0)
        print(io, ":")
        show(IOContext(io, :typeinfo => eltype(S)), S)
    end
end
