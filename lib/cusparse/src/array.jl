# sparse arrays
#
# The ordinary formats (vectors, CSR, CSC and COO matrices) are GPUArrays' generic sparse
# types with `CuVector` storage, so every operation GPUArrays implements works on them.
# cuSPARSE-specific formats (BSR, batched CSR) are defined here.

export CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR, CuSparseMatrixCOO,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseArrayCSR,
       CuSparseVector,
       CuSparseVecOrMat

using LinearAlgebra: BlasFloat
using SparseArrays
using SparseArrays: nonzeroinds, nonzeros, rowvals, getcolptr, dimlub
using GPUArrays: GPUSparseVector, GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO
import CUDACore: CuArray


## canonical formats

"""
    CuSparseVector{Tv,Ti}

A sparse vector with values of type `Tv` and indices of type `Ti` stored in `CuVector`s:
GPUArrays' `GPUSparseVector` with CUDA storage.
"""
const CuSparseVector{Tv,Ti} = GPUSparseVector{Tv,Ti,<:CuVector{Ti},<:CuVector{Tv}}

"""
    CuSparseMatrixCSR{Tv,Ti}

A sparse matrix in compressed sparse row format stored in `CuVector`s: GPUArrays'
`GPUSparseMatrixCSR` with CUDA storage. Most cuSPARSE routines work best with this format.
"""
const CuSparseMatrixCSR{Tv,Ti} = GPUSparseMatrixCSR{Tv,Ti,<:CuVector{Ti},<:CuVector{Tv}}

"""
    CuSparseMatrixCSC{Tv,Ti}

A sparse matrix in compressed sparse column format stored in `CuVector`s: GPUArrays'
`GPUSparseMatrixCSC` with CUDA storage, the layout of `SparseMatrixCSC`. This is what `cu`
makes of a `SparseMatrixCSC`.
"""
const CuSparseMatrixCSC{Tv,Ti} = GPUSparseMatrixCSC{Tv,Ti,<:CuVector{Ti},<:CuVector{Tv}}

"""
    CuSparseMatrixCOO{Tv,Ti}

A sparse matrix in coordinate format, sorted by row, stored in `CuVector`s: GPUArrays'
`GPUSparseMatrixCOO` with CUDA storage.
"""
const CuSparseMatrixCOO{Tv,Ti} = GPUSparseMatrixCOO{Tv,Ti,<:CuVector{Ti},<:CuVector{Tv}}

const CuCanonicalMatrix{Tv,Ti} =
    Union{CuSparseMatrixCSR{Tv,Ti},CuSparseMatrixCSC{Tv,Ti},CuSparseMatrixCOO{Tv,Ti}}
const CuCanonicalSparse{Tv,Ti} = Union{CuSparseVector{Tv,Ti},CuCanonicalMatrix{Tv,Ti}}


## cuSPARSE-specific formats

abstract type AbstractCuVendorSparseMatrix{Tv,Ti} <: GPUArrays.AbstractGPUSparseMatrix{Tv,Ti} end

"""
    CuSparseMatrixBSR

Container to hold sparse matrices in block compressed sparse row (BSR) format on
the GPU. BSR format is also used in Intel MKL, and is suited to matrices that are
"block" sparse - rare blocks of non-sparse regions.
"""
mutable struct CuSparseMatrixBSR{Tv, Ti} <: AbstractCuVendorSparseMatrix{Tv, Ti}
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

function CUDACore.unsafe_free!(xs::CuSparseMatrixBSR)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(nonzeros(xs))
    return
end

mutable struct CuSparseArrayCSR{Tv, Ti, N} <: GPUArrays.AbstractGPUSparseArray{Tv, Ti, N}
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

function CUDACore.unsafe_free!(xs::CuSparseArrayCSR)
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
const CuSparseMatrix{Tv,Ti} = Union{CuSparseMatrixCSR{Tv,Ti},CuSparseMatrixCSC{Tv,Ti},
                                   CuSparseMatrixCOO{Tv,Ti},CuSparseMatrixBSR{Tv,Ti}}

# (the formats are concrete GPUArrays types now, so this can no longer be subtyped)
const AbstractCuSparseMatrix{Tv,Ti} =
    Union{CuCanonicalMatrix{Tv,Ti},AbstractCuVendorSparseMatrix{Tv,Ti}}

const CuSparseVecOrMat = Union{CuSparseVector,CuSparseMatrix}
const CuSparseMatrixAdjOrTrans =
    Union{Transpose{<:Any,<:CuCanonicalMatrix},Adjoint{<:Any,<:CuCanonicalMatrix}}
const CuSparseMatrixConvertible = Union{CuSparseMatrix,CuSparseMatrixAdjOrTrans}


## constructors

# From buffers: like SparseArrays' raw constructors, these borrow the buffers (unless
# types need to be converted) and do not verify the structure. The index buffers of a
# matrix share an element type, so differing ones are promoted.
uniform_indices(p::CuVector{Ti}, i::CuVector{Ti}) where {Ti} = (p, i)
function uniform_indices(p::CuVector, i::CuVector)
    Ti = promote_type(eltype(p), eltype(i))
    (convert(CuVector{Ti}, p), convert(CuVector{Ti}, i))
end
for (alias, generic) in ((:CuSparseMatrixCSR, :GPUSparseMatrixCSR),
                         (:CuSparseMatrixCSC, :GPUSparseMatrixCSC),
                         (:CuSparseMatrixCOO, :GPUSparseMatrixCOO))
    @eval begin
        $alias(p::CuVector{<:Integer}, i::CuVector{<:Integer}, v::CuVector,
               dims::NTuple{2,Integer}) =
            $generic(uniform_indices(p, i)..., v, dims)
        $alias{Tv}(p::CuVector{<:Integer}, i::CuVector{<:Integer}, v::CuVector,
                   dims::NTuple{2,Integer}) where {Tv} =
            $generic(uniform_indices(p, i)..., convert(CuVector{Tv}, v), dims)
        $alias{Tv,Ti}(p::CuVector{<:Integer}, i::CuVector{<:Integer}, v::CuVector,
                      dims::NTuple{2,Integer}) where {Tv,Ti} =
            $generic(convert(CuVector{Ti}, p), convert(CuVector{Ti}, i),
                     convert(CuVector{Tv}, v), dims)
    end
end
CuSparseVector(i::CuVector{<:Integer}, v::CuVector, n::Integer) = GPUSparseVector(i, v, n)
CuSparseVector{Tv}(i::CuVector{<:Integer}, v::CuVector, n::Integer) where {Tv} =
    GPUSparseVector(i, convert(CuVector{Tv}, v), n)
CuSparseVector{Tv,Ti}(i::CuVector{<:Integer}, v::CuVector, n::Integer) where {Tv,Ti} =
    GPUSparseVector(convert(CuVector{Ti}, i), convert(CuVector{Tv}, v), n)

# COO matrices can also be constructed without dimensions, or from the first `n` entries
# of the buffers
CuSparseMatrixCOO(I::CuVector{<:Integer}, J::CuVector{<:Integer}, V::CuVector) =
    CuSparseMatrixCOO(I, J, V, (dimlub(I), dimlub(J)))
CuSparseMatrixCOO{Tv}(I::CuVector{<:Integer}, J::CuVector{<:Integer}, V::CuVector) where {Tv} =
    CuSparseMatrixCOO{Tv}(I, J, V, (dimlub(I), dimlub(J)))
CuSparseMatrixCOO{Tv,Ti}(I::CuVector{<:Integer}, J::CuVector{<:Integer}, V::CuVector) where {Tv,Ti} =
    CuSparseMatrixCOO{Tv,Ti}(I, J, V, (dimlub(I), dimlub(J)))
function coo_prefix(I, J, V, n::Integer)
    0 <= n <= min(length(I), length(J), length(V)) ||
        throw(ArgumentError("$n stored entries exceed the length of the buffers"))
    n == length(I) == length(J) == length(V) && return I, J, V
    return I[1:n], J[1:n], V[1:n]
end
CuSparseMatrixCOO(I::CuVector{<:Integer}, J::CuVector{<:Integer}, V::CuVector,
                  dims::NTuple{2,Integer}, n::Integer) =
    CuSparseMatrixCOO(coo_prefix(I, J, V, n)..., dims)
CuSparseMatrixCOO{Tv}(I::CuVector{<:Integer}, J::CuVector{<:Integer}, V::CuVector,
                      dims::NTuple{2,Integer}, n::Integer) where {Tv} =
    CuSparseMatrixCOO{Tv}(coo_prefix(I, J, V, n)..., dims)
CuSparseMatrixCOO{Tv,Ti}(I::CuVector{<:Integer}, J::CuVector{<:Integer}, V::CuVector,
                         dims::NTuple{2,Integer}, n::Integer) where {Tv,Ti} =
    CuSparseMatrixCOO{Tv,Ti}(coo_prefix(I, J, V, n)..., dims)

# From other arrays: move the input to CUDA, then convert on the device. As in CUDA.jl 6,
# host arrays and dense GPU arrays get `Cint` indices (unless the dimensions or number of
# entries do not fit), and sparse GPU arrays keep their index type. `{Tv,Ti}` selects the
# index type explicitly.
cuda_indtype(A::GPUArrays.AbstractGPUSparseArray) = SparseArrays.indtype(A)
cuda_indtype(A::Union{Transpose,Adjoint}) = cuda_indtype(parent(A))
function cuda_indtype(A::AbstractArray)
    n = A isa AbstractSparseArray ? nnz(A) : length(A)
    max(size(A)..., n + 1) <= typemax(Cint) && return Cint
    return A isa AbstractSparseArray ? SparseArrays.indtype(A) : Int
end
for (alias, generic) in ((:CuSparseVector, :GPUSparseVector),
                         (:CuSparseMatrixCSR, :GPUSparseMatrixCSR),
                         (:CuSparseMatrixCSC, :GPUSparseMatrixCSC),
                         (:CuSparseMatrixCOO, :GPUSparseMatrixCOO))
    @eval begin
        # (conversions to the type of the input return it, as in CUDA.jl 6)
        $alias(A::$alias; kwargs...) = A
        $alias{Tv}(A::$alias{Tv}; kwargs...) where {Tv} = A
        $alias{Tv,Ti}(A::$alias{Tv,Ti}; kwargs...) where {Tv,Ti} = A
        $alias(A::AbstractArray; kwargs...) = $alias{eltype(A),cuda_indtype(A)}(A; kwargs...)
        $alias{Tv}(A::AbstractArray; kwargs...) where {Tv} =
            $alias{Tv,cuda_indtype(A)}(A; kwargs...)
        function $alias{Tv,Ti}(A::AbstractArray; index::SparseChar='O', sorted::Bool=true) where {Tv,Ti}
            # (the keywords of CUDA.jl 6. Sparse arrays are one-based, but zero-based
            #  buffers can still be converted between formats.)
            if index == 'Z' && A isa CuCanonicalSparse
                return shift_indices($generic{Tv,Ti}(shift_indices(A, 1)), -1)
            end
            index == 'O' || throw(ArgumentError("index must be 'O' or (for sparse GPU arrays) 'Z'"))
            $generic{Tv,Ti}(adapt(CuArray, A))
        end
    end
end

# the same array with `d` added to its indices and pointers
function shift_indices(A::CuCanonicalSparse, d::Integer)
    buffers = GPUArrays.sparse_buffers(A)
    shifted = map(b -> b .+ eltype(b)(d), Base.front(buffers))
    typeof(A).name.wrapper(shifted..., last(buffers), A isa CuSparseVector ? length(A) : size(A))
end

CuSparseArrayCSR(p::CuArray, i::CuArray, v::CuArray{T}, dims::NTuple{N,Integer}) where {T,N} =
    CuSparseArrayCSR{T,eltype(i),N}(p, i, v, dims)
CuSparseArrayCSR{T}(p::CuArray, i::CuArray, v::CuArray, dims::NTuple{N,Integer}) where {T,N} =
    CuSparseArrayCSR{T,eltype(i),N}(p, i, v, dims)

CuSparseMatrixBSR{T}(p::CuVector, i::CuVector, v::CuVector, dims, b, dir, n) where {T} =
    CuSparseMatrixBSR{T,eltype(i)}(p, i, v, dims, b, dir, n)
CuSparseMatrixBSR(p::CuVector, i::CuVector, v::CuVector{T}, b, dir, n, dims) where {T} =
    CuSparseMatrixBSR{T,eltype(i)}(p, i, v, dims, b, dir, n)
# (converting returns the input if the types match, and copies every buffer otherwise)
CuSparseMatrixBSR{T,Ti}(A::CuSparseMatrixBSR{T,Ti}) where {T,Ti} = A
function CuSparseMatrixBSR{T,Ti}(A::CuSparseMatrixBSR) where {T,Ti}
    copied(::Type{S}, x) where {S} = eltype(x) === S ? copy(x) : convert(CuVector{S}, x)
    CuSparseMatrixBSR{T,Ti}(copied(Ti, A.rowPtr), copied(Ti, A.colVal), copied(T, A.nzVal),
                            size(A), A.blockDim, A.dir, A.nnzb)
end
CuSparseMatrixBSR{T}(A::CuSparseMatrixBSR{<:Any,Ti}) where {T,Ti} = CuSparseMatrixBSR{T,Ti}(A)
CuSparseMatrixBSR{T}(A::CuSparseMatrixBSR{T}) where {T} = A
CuSparseMatrixBSR{T,Ti}(A::SparseMatrixCSC, b) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(CuSparseMatrixBSR(CuSparseMatrixCSR{T,Cint}(A), b))
# (the BSR routines of cuSPARSE only take 32-bit indices, so that is the default)
CuSparseMatrixBSR{T}(A::SparseMatrixCSC, b) where {T} = CuSparseMatrixBSR{T,Cint}(A, b)
CuSparseMatrixBSR(A::SparseMatrixCSC{T}, b) where {T} = CuSparseMatrixBSR{T,Cint}(A, b)
# other host sparse matrices (e.g. from package extensions) go through CSR
CuSparseMatrixBSR{T}(A::AbstractSparseMatrix, b) where {T} =
    CuSparseMatrixBSR(CuSparseMatrixCSR{T,Cint}(A), b)
CuSparseMatrixBSR(A::AbstractSparseMatrix{T}, b) where {T} = CuSparseMatrixBSR{T}(A, b)


## `cu` and `adapt`

# Host sparse arrays move to CSC matrices (or vectors) with `Cint` indices, unless they do
# not fit, as in CUDA.jl 6. `cu` additionally applies the dense element-type rules (Float64
# becomes Float32), and `adapt(CuArray{T}, S)` converts the values to `T`. (GPUArrays'
# generic rules would keep the index type; GPU sparse arrays are adapted by those.)
cu_indtype(A::AbstractSparseArray) =
    max(size(A)..., nnz(A) + 1) <= typemax(Cint) ? Cint : SparseArrays.indtype(A)
cu_indices(proto::CuVector, ::Type{Ti}, inds::Vector) where {Ti} =
    copyto!(similar(proto, Ti, length(inds)), convert(Vector{Ti}, inds))

# the values of `A` adapted for `to`, as a vector (the target may be e.g. `CuMatrix`)
cu_values(to::CUDACore.CuArrayKernelAdaptor, A) = adapt(to, nonzeros(A))
cu_values(to::Type{<:CuArray}, A) =
    CuArray{CUDACore.adapt_eltype(to, A), 1, CUDACore.adapt_memory(to, A)}(nonzeros(A))

const CuAdaptor = Union{CUDACore.CuArrayKernelAdaptor, Type{<:CuArray}}
function Adapt.adapt_structure(to::CuAdaptor, A::SparseMatrixCSC)
    Ti = cu_indtype(A)
    nzVal = cu_values(to, A)
    GPUSparseMatrixCSC(cu_indices(nzVal, Ti, getcolptr(A)), cu_indices(nzVal, Ti, rowvals(A)),
                       nzVal, size(A))
end
function Adapt.adapt_structure(to::CuAdaptor, x::SparseVector)
    Ti = cu_indtype(x)
    nzVal = cu_values(to, x)
    GPUSparseVector(cu_indices(nzVal, Ti, nonzeroinds(x)), nzVal, length(x))
end

# densify on the device (the generic constructors would index every element)
for (N, S) in ((1, :CuSparseVector), (2, :CuCanonicalMatrix))
    @eval begin
        CuArray{T,$N,M}(A::$S) where {T,M} = copyto!(CuArray{T,$N,M}(undef, size(A)), A)
        CuArray{T,$N,M}(A::$S{T}) where {T,M} = copyto!(CuArray{T,$N,M}(undef, size(A)), A)
        CuArray{T,$N}(A::$S) where {T} = copyto!(CuArray{T,$N}(undef, size(A)), A)
        CuArray{T,$N}(A::$S{T}) where {T} = copyto!(CuArray{T,$N}(undef, size(A)), A)
        CuArray{T}(A::$S) where {T} = CuArray{T,$N}(A)
        CuArray(A::$S) = CuArray{eltype(A),$N}(A)
    end
end

# CUDA.jl 6 stored the number of entries, and named the indices of a vector `iPtr`
@inline function Base.getproperty(x::CuSparseVector, s::Symbol)
    if s === :iPtr
        Base.depwarn("`x.iPtr` is deprecated, use `SparseArrays.nonzeroinds(x)` instead.", :iPtr)
        return getfield(x, :nzInd)
    elseif s === :nnz
        Base.depwarn("`x.nnz` is deprecated, use `nnz(x)` instead.", :nnz)
        return convert(SparseArrays.indtype(x), nnz(x))
    end
    return getfield(x, s)
end
@inline function Base.getproperty(A::CuCanonicalMatrix, s::Symbol)
    if s === :nnz
        Base.depwarn("`A.nnz` is deprecated, use `nnz(A)` instead.", :nnz)
        return convert(SparseArrays.indtype(A), nnz(A))
    end
    return getfield(A, s)
end

"""
    unsafe_free!(A::CuSparseVector)
    unsafe_free!(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO})

Release the buffers of a sparse array. Constructors from buffers borrow them, so this is
only safe when no other array (or view) uses any of them.
"""
function CUDACore.unsafe_free!(A::CuCanonicalSparse)
    foreach(unsafe_free!, GPUArrays.sparse_buffers(A))
    return
end


## BSR and batched CSR array interface

Base.size(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}) = A.dims
function Base.size(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}, d::Integer)
    d >= 1 || throw(ArgumentError("dimension must be ≥ 1, got $d"))
    return d <= ndims(A) ? A.dims[d] : 1
end
Base.length(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}) = prod(A.dims)
SparseArrays.nnz(A::CuSparseMatrixBSR) = A.nnzb * A.blockDim * A.blockDim
SparseArrays.nnz(A::CuSparseArrayCSR) = A.nnz
SparseArrays.nonzeros(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}) = A.nzVal

Base.similar(A::CuSparseMatrixBSR{Tv,Ti}) where {Tv,Ti} =
    CuSparseMatrixBSR{Tv,Ti}(copy(A.rowPtr), copy(A.colVal), similar(nonzeros(A)), size(A),
                             A.blockDim, A.dir, A.nnzb)
Base.similar(A::CuSparseMatrixBSR{<:Any,Ti}, ::Type{T}) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(copy(A.rowPtr), copy(A.colVal), similar(nonzeros(A), T), size(A),
                            A.blockDim, A.dir, A.nnzb)
Base.similar(A::CuSparseMatrixBSR, ::Type{T}, ::Type{Ti}) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(convert(CuVector{Ti}, copy(A.rowPtr)), convert(CuVector{Ti}, copy(A.colVal)),
                            similar(nonzeros(A), T), size(A), A.blockDim, A.dir, A.nnzb)
# with dimensions: a BSR matrix without stored blocks
Base.similar(A::CuSparseMatrixBSR, ::Type{T}, ::Type{Ti}, m::Int, n::Int) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(fill!(CuVector{Ti}(undef, cld(m, A.blockDim) + 1), one(Ti)),
                            CuVector{Ti}(undef, 0), CuVector{T}(undef, 0), (m, n), A.blockDim,
                            A.dir, 0)
Base.similar(A::CuSparseMatrixBSR{<:Any,Ti}, ::Type{T}, m::Int, n::Int) where {T,Ti} =
    similar(A, T, Ti, m, n)
Base.similar(A::CuSparseMatrixBSR, m::Int, n::Int) = similar(A, eltype(A), m, n)
Base.similar(A::CuSparseMatrixBSR, ::Type{T}, dims::Tuple{Int,Int}) where {T} = similar(A, T, dims...)
Base.similar(A::CuSparseMatrixBSR, ::Type{T}, ::Type{Ti}, dims::Tuple{Int,Int}) where {T,Ti} =
    similar(A, T, Ti, dims...)
Base.similar(A::CuSparseMatrixBSR, dims::Tuple{Int,Int}) = similar(A, dims...)
Base.similar(A::CuSparseArrayCSR) =
    CuSparseArrayCSR(copy(A.rowPtr), copy(A.colVal), similar(nonzeros(A)), size(A))

Base.copy(A::CuSparseMatrixBSR) =
    CuSparseMatrixBSR{eltype(A),eltype(A.rowPtr)}(copy(A.rowPtr), copy(A.colVal),
                                                  copy(nonzeros(A)), size(A), A.blockDim,
                                                  A.dir, A.nnzb)
Base.copy(A::CuSparseArrayCSR) =
    CuSparseArrayCSR(copy(A.rowPtr), copy(A.colVal), copy(nonzeros(A)), size(A))

function Base.copyto!(dst::CuSparseMatrixBSR, src::CuSparseMatrixBSR)
    size(dst) == size(src) ||
        throw(DimensionMismatch("destination has size $(size(dst)), source has size $(size(src))"))
    dst === src && return dst
    for (d, s) in ((dst.rowPtr, src.rowPtr), (dst.colVal, src.colVal), (dst.nzVal, src.nzVal))
        resize!(d, length(s))
        copyto!(d, s)
    end
    dst.blockDim = src.blockDim
    dst.dir = src.dir
    dst.nnzb = src.nnzb
    return dst
end

function Base.getindex(A::CuSparseMatrixBSR{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    i0_block, i0_idx = fldmod1(i0, A.blockDim)
    i1_block, i1_idx = fldmod1(i1, A.blockDim)
    block_idx = A.dir == 'R' ? (i0_idx-1)*A.blockDim+i1_idx : (i1_idx-1)*A.blockDim+i0_idx
    c1 = Int(A.rowPtr[i0_block])
    c2 = Int(A.rowPtr[i0_block+1]-1)
    for k in c1:c2
        if A.colVal[k] == i1_block
            return nonzeros(A)[(k-1)*A.blockDim*A.blockDim+block_idx]
        end
    end
    return zero(T)
end
Base.getindex(A::CuSparseMatrixBSR, ::Colon) = CuSparseMatrixCSC(A)[:]
Base.getindex(A::CuSparseMatrixBSR, I::AbstractUnitRange) = CuSparseMatrixCSC(A)[I]
Base.getindex(A::CuSparseMatrixBSR, I::Union{Integer,AbstractUnitRange,Colon},
              J::Union{AbstractUnitRange,Colon}) = CuSparseMatrixCSR(A)[I, J]
Base.getindex(A::CuSparseMatrixBSR, I::Union{AbstractUnitRange,Colon}, j::Integer) =
    CuSparseMatrixCSC(A)[I, j]

function Base.getindex(A::CuSparseArrayCSR{Tv, Ti, N}, ::Colon, ::Colon, idxs::Integer...) where {Tv, Ti, N}
    @boundscheck checkbounds(A, :, :, idxs...)
    CuSparseMatrixCSR(A.rowPtr[:,idxs...], A.colVal[:,idxs...], nonzeros(A)[:,idxs...], size(A)[1:2])
end

function Base.getindex(A::CuSparseArrayCSR{Tv, Ti, N}, i0::Integer, i1::Integer, idxs::Integer...) where {Tv, Ti, N}
    @boundscheck checkbounds(A, i0, i1, idxs...)
    CuSparseMatrixCSR(A.rowPtr[:,idxs...], A.colVal[:,idxs...], nonzeros(A)[:,idxs...], size(A)[1:2])[i0, i1]
end

# BSR matrices go through CSR for host transfers, densification and display
SparseArrays.SparseMatrixCSC(A::CuSparseMatrixBSR) = SparseMatrixCSC(CuSparseMatrixCSR(A))
Base.Array(A::CuSparseMatrixBSR) = Array(CuSparseMatrixCSR(A))
Base.collect(A::CuSparseMatrixBSR) = Array(A)
CuArray(A::CuSparseMatrixBSR) = CuArray(CuSparseMatrixCSR(A))
CuArray{T}(A::CuSparseMatrixBSR) where {T} = CuArray{T}(CuSparseMatrixCSR(A))
Adapt.adapt_storage(::Type{Array}, A::CuSparseMatrixBSR) = SparseMatrixCSC(A)

Base.show(io::IO, A::CuSparseMatrixBSR) = show(io, SparseMatrixCSC(A))
function Base.show(io::IO, ::MIME"text/plain", A::CuSparseMatrixBSR)
    xnnz = nnz(A)
    m, n = size(A)
    print(io, m, "×", n, " ", typeof(A), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if !(m == 0 || n == 0)
        println(io, ":")
        Base.print_array(IOContext(io, :typeinfo => eltype(A)), SparseMatrixCSC(A))
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseArrayCSR)
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


## device representations of the cuSPARSE-specific formats

# (the canonical formats are adapted field by field by GPUArrays)
struct CuSparseDeviceMatrixBSR{Tv,Ti,Vi,Vv} <: SparseArrays.AbstractSparseMatrix{Tv,Ti}
    rowPtr::Vi
    colVal::Vi
    nzVal::Vv
    dims::Dims{2}
    blockDim::Ti
    dir::Char
    nnz::Ti
end
struct CuSparseDeviceArrayCSR{Tv,Ti,N,Vi,Vv} <: SparseArrays.AbstractSparseArray{Tv,Ti,N}
    rowPtr::Vi
    colVal::Vi
    nzVal::Vv
    dims::Dims{N}
    nnz::Ti
end
Base.size(A::Union{CuSparseDeviceMatrixBSR,CuSparseDeviceArrayCSR}) = A.dims
SparseArrays.nnz(A::CuSparseDeviceMatrixBSR) = length(A.nzVal)
SparseArrays.nnz(A::CuSparseDeviceArrayCSR) = A.nnz
SparseArrays.nonzeros(A::Union{CuSparseDeviceMatrixBSR,CuSparseDeviceArrayCSR}) = A.nzVal

function Adapt.adapt_structure(to::CUDACore.KernelAdaptor, A::CuSparseMatrixBSR{T,Ti}) where {T,Ti}
    p, i, v = adapt(to, A.rowPtr), adapt(to, A.colVal), adapt(to, A.nzVal)
    CuSparseDeviceMatrixBSR{T,Ti,typeof(p),typeof(v)}(p, i, v, size(A), A.blockDim, A.dir, A.nnzb)
end
function Adapt.adapt_structure(to::CUDACore.KernelAdaptor, A::CuSparseArrayCSR{T,Ti,N}) where {T,Ti,N}
    p, i, v = adapt(to, A.rowPtr), adapt(to, A.colVal), adapt(to, A.nzVal)
    CuSparseDeviceArrayCSR{T,Ti,N,typeof(p),typeof(v)}(p, i, v, size(A), A.nnz)
end
