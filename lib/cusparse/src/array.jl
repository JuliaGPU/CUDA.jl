# Ordinary sparse arrays use GPUArrays' canonical storage. BSR and batching retain their
# backend formats; descriptors and unfinished vendor outputs are separate resources.
using SparseArrays
using SparseArrays: nonzeroinds, nonzeros, rowvals, getcolptr
using LinearAlgebra: BlasFloat, AdjOrTrans
import CUDACore: CuArray
using GPUArrays: GPUSparseVector, GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO

export CuSparseVector, CuSparseMatrixCSR, CuSparseMatrixCSC, CuSparseMatrixCOO,
       CuSparseMatrixBSR, CuSparseArrayCSR, CuSparseMatrix, CuSparseVecOrMat,
       AbstractCuSparseMatrix

const CuSparseVector{Tv,Ti} = GPUSparseVector{Tv,Ti,<:CuVector{Ti},<:CuVector{Tv}}
const CuSparseMatrixCSR{Tv,Ti} = GPUSparseMatrixCSR{Tv,Ti,<:CuVector{<:Integer},<:CuVector{Ti},<:CuVector{Tv}}
const CuSparseMatrixCSC{Tv,Ti} = GPUSparseMatrixCSC{Tv,Ti,<:CuVector{<:Integer},<:CuVector{Ti},<:CuVector{Tv}}
const CuSparseMatrixCOO{Tv,Ti} = GPUSparseMatrixCOO{Tv,Ti,<:CuVector{Ti},<:CuVector{Tv}}
const CuCanonicalMatrix{Tv,Ti} = Union{CuSparseMatrixCSR{Tv,Ti},CuSparseMatrixCSC{Tv,Ti},CuSparseMatrixCOO{Tv,Ti}}
const CuCanonicalSparse{Tv,Ti} = Union{CuSparseVector{Tv,Ti},CuCanonicalMatrix{Tv,Ti}}

abstract type _AbstractCuVendorSparseMatrix{Tv,Ti} <: GPUArrays.AbstractGPUSparseMatrix{Tv,Ti} end

mutable struct CuSparseMatrixBSR{Tv, Ti} <: _AbstractCuVendorSparseMatrix{Tv, Ti}
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
        all(>=(0),dims) && blockDim>0 || throw(ArgumentError("invalid BSR dimensions or block size"))
        dir in ('R','C') || throw(ArgumentError("invalid BSR block direction"))
        mb,nb=cld.(dims,blockDim)
        GPUArrays.check_sparse_dims(Ti,(mb,nb,Int(blockDim)))
        GPUArrays.check_sparse_nnz(Ti,nnz)
        length(rowPtr)==Base.checked_add(mb,1) && length(colVal)==nnz &&
            length(nzVal)==Base.checked_mul(nnz,Base.checked_mul(blockDim,blockDim)) ||
            throw(ArgumentError("BSR buffer lengths do not match block dimensions and count"))
        GPUArrays.check_sparse_values(Tv,nzVal)
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


const CuSparseMatrix{Tv,Ti} = Union{CuSparseMatrixCSR{Tv,Ti},CuSparseMatrixCSC{Tv,Ti},
                                   CuSparseMatrixCOO{Tv,Ti},CuSparseMatrixBSR{Tv,Ti}}
# Kept for dispatch, but this UnionAll alias can no longer be subtyped.
const AbstractCuSparseMatrix{Tv,Ti} = Union{CuCanonicalMatrix{Tv,Ti},_AbstractCuVendorSparseMatrix{Tv,Ti}}
const CuSparseVecOrMat = Union{CuSparseVector,CuSparseMatrix}
const CuSparseMatrixAdjOrTrans = Union{Transpose{<:Any,<:CuCanonicalMatrix},Adjoint{<:Any,<:CuCanonicalMatrix}}
const CuSparseMatrixConvertible = Union{CuSparseMatrix,CuSparseMatrixAdjOrTrans}

# Raw constructors borrow canonical buffers and infer the two integer widths. An explicit
# {Tv,Ti} requests uniform indices; value-only forms preserve existing storage widths.
for (alias, generic) in ((:CuSparseMatrixCSR,:GPUSparseMatrixCSR),
                         (:CuSparseMatrixCSC,:GPUSparseMatrixCSC),
                         (:CuSparseMatrixCOO,:GPUSparseMatrixCOO))
    @eval begin
        $alias(p::CuVector, i::CuVector, v::CuVector, dims::NTuple{2,Integer}) =
            $generic(p, i, v, dims)
        $alias{Tv}(p::CuVector, i::CuVector, v::CuVector, dims::NTuple{2,Integer}) where {Tv} =
            $generic(p, i, checked_values(Tv, v), dims)
        $alias{Tv,Ti}(p::CuVector, i::CuVector, v::CuVector, dims::NTuple{2,Integer}) where {Tv,Ti} =
            GPUArrays.with_eltypes($generic(p, i, v, dims), Tv, Ti)
        $alias(A::GPUArrays.GPUSparseArray;index::SparseChar='O') =
            (index=='O' || throw(ArgumentError("canonical arrays use one-based indices"));$generic(adapt(CuArray,A)))
        $alias{Tv}(A::GPUArrays.GPUSparseArray;index::SparseChar='O') where {Tv} =
            (index=='O' || throw(ArgumentError("canonical arrays use one-based indices"));$generic{Tv}(adapt(CuArray,A)))
        $alias{Tv,Ti}(A::GPUArrays.GPUSparseArray;index::SparseChar='O') where {Tv,Ti} =
            (index=='O' || throw(ArgumentError("canonical arrays use one-based indices"));$generic{Tv,Ti}(adapt(CuArray,A)))
        $alias(A::AbstractArray) = $alias{eltype(A),Cint}(A)
        $alias{Tv}(A::AbstractArray) where {Tv} = $alias{Tv,Cint}(A)
        $alias{Tv,Ti}(A::AbstractArray) where {Tv,Ti} =
            $generic{Tv,Ti}(adapt(CuArray, A))
    end
end
checked_values(::Type{T}, v) where {T} =
    (GPUArrays.check_sparse_values(T, v); convert(CuVector{T}, v))

CuSparseVector(i::CuVector, v::CuVector, n::Integer) = GPUSparseVector(i, v, n)
CuSparseVector{T}(i::CuVector, v::CuVector, n::Integer) where {T} =
    GPUSparseVector(i, checked_values(T, v), n)
CuSparseVector{T,Ti}(i::CuVector, v::CuVector, n::Integer) where {T,Ti} =
    GPUArrays.with_eltypes(GPUSparseVector(i, v, n), T, Ti)
CuSparseVector(A::GPUArrays.GPUSparseArray) = GPUSparseVector(adapt(CuArray, A))
CuSparseVector{T}(A::GPUArrays.GPUSparseArray) where {T} = GPUSparseVector{T}(adapt(CuArray, A))
CuSparseVector{T,Ti}(A::GPUArrays.GPUSparseArray) where {T,Ti} = GPUSparseVector{T,Ti}(adapt(CuArray, A))
CuSparseVector(A::AbstractArray) = CuSparseVector{eltype(A),Cint}(A)
CuSparseVector{T}(A::AbstractArray) where {T} = CuSparseVector{T,Cint}(A)
CuSparseVector{T,Ti}(A::AbstractArray) where {T,Ti} = GPUSparseVector{T,Ti}(adapt(CuArray, A))

CuSparseMatrixCOO(I::CuVector,J::CuVector,V::CuVector) =
    CuSparseMatrixCOO(I,J,V,(isempty(I) ? 0 : Int(maximum(I)),isempty(J) ? 0 : Int(maximum(J))))
CuSparseMatrixCOO{T}(I::CuVector,J::CuVector,V::CuVector) where {T} =
    GPUArrays.with_values(CuSparseMatrixCOO(I,J,V),T)
CuSparseMatrixCOO{T,Ti}(I::CuVector,J::CuVector,V::CuVector) where {T,Ti} =
    GPUArrays.with_eltypes(CuSparseMatrixCOO(I,J,V),T,Ti)

# Legacy explicit stored counts are logical lengths, never spare capacity.
function _coo_prefix(I, J, V, dims, n::Integer)
    0 <= n <= min(length(I), length(J), length(V)) ||
        throw(ArgumentError("stored count exceeds supplied COO buffers"))
    if n != length(I) || n != length(J) || n != length(V)
        return GPUSparseMatrixCOO(copy(view(I,1:n)), copy(view(J,1:n)), copy(view(V,1:n)), dims)
    end
    return GPUSparseMatrixCOO(I, J, V, dims)
end
CuSparseMatrixCOO(I::CuVector, J::CuVector, V::CuVector, dims::NTuple{2,Integer}, n::Integer) =
    _coo_prefix(I,J,V,dims,n)
CuSparseMatrixCOO{T,Ti}(I::CuVector, J::CuVector, V::CuVector, dims::NTuple{2,Integer}, n::Integer) where {T,Ti} =
    GPUArrays.with_eltypes(_coo_prefix(I,J,V,dims,n),T,Ti)
CuSparseMatrixCOO{T}(I::CuVector, J::CuVector, V::CuVector, dims::NTuple{2,Integer}, n::Integer) where {T} =
    GPUArrays.with_values(_coo_prefix(I,J,V,dims,n),T)

# cu retains CSC/Int32 storage and its dense precision/memory policy. Structural adapt
# to an array type is supplied by GPUArrays and preserves the input's integer widths.
for S in (:SparseMatrixCSC,:SparseVector,:GPUSparseMatrixCSR,:GPUSparseMatrixCSC,
          :GPUSparseMatrixCOO,:GPUSparseVector)
    @eval function Adapt.adapt_structure(to::Type{<:CuArray},A::$S)
        values=nonzeros(A)
        storage=CuArray{CUDACore.adapt_eltype(to,values),1,CUDACore.adapt_memory(to,values)}
        GPUArrays.adapt_sparse_storage(storage,A)
    end
end
for S in (:SparseMatrixCSC,:SparseVector)
    @eval function Adapt.adapt_structure(to::CUDACore.CuArrayKernelAdaptor, A::$S)
        values = adapt(to, nonzeros(A))
        B = adapt(typeof(values), A)
        return GPUArrays.sparse_format_type(B){eltype(values),Cint}(B)
    end
end

# Dense construction stays on the device, including explicit memory modes.
for (N, S) in ((1,:CuSparseVector),(2,:CuCanonicalMatrix))
    @eval begin
        CuArray{T,$N,M}(A::$S) where {T,M} = copyto!(CuArray{T,$N,M}(undef,size(A)),A)
        CuArray{T,$N,M}(A::$S{T}) where {T,M} = copyto!(CuArray{T,$N,M}(undef,size(A)),A)
        CuArray{T,$N}(A::$S) where {T} = copyto!(CuArray{T,$N}(undef,size(A)),A)
        CuArray{T,$N}(A::$S{T}) where {T} = copyto!(CuArray{T,$N}(undef,size(A)),A)
        CuArray{T}(A::$S) where {T} = CuArray{T,$N}(A)
        CuArray(A::$S) = CuArray{eltype(A),$N}(A)
    end
end

"""
    unsafe_free!(A::CuCanonicalSparse)

Release all component buffers. The caller must own every buffer exclusively, with no
borrowed views, device representations or retained vendor analysis still using them.
Raw sparse constructors borrow storage, so this operation is not safe for arbitrary inputs.
"""
function CUDACore.unsafe_free!(A::CuCanonicalSparse)
    foreach(unsafe_free!, GPUArrays.sparse_buffers(A))
    return nothing
end

Base.size(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}) = A.dims
Base.size(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR},d::Integer) = d<1 ?
    throw(ArgumentError("dimension must be ≥ 1, got $d")) : d<=ndims(A) ? size(A)[d] : 1
Base.length(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}) = prod(A.dims)
SparseArrays.nnz(A::CuSparseMatrixBSR) = length(A.nzVal)
SparseArrays.nnz(A::CuSparseArrayCSR) = A.nnz
SparseArrays.nonzeros(A::Union{CuSparseMatrixBSR,CuSparseArrayCSR}) = A.nzVal
Base.getindex(A::CuSparseMatrixBSR,::Colon) = CuSparseMatrixCSC(A)[:]
Base.getindex(A::CuSparseMatrixBSR,I::AbstractRange{<:Integer}) = CuSparseMatrixCSC(A)[:][I]
Base.getindex(A::CuSparseMatrixBSR,I::Union{Integer,AbstractRange,Colon},J::Union{AbstractRange,Colon}) =
    CuSparseMatrixCSR(A)[I,J]
Base.getindex(A::CuSparseMatrixBSR,I::Union{AbstractRange,Colon},J::Integer) = CuSparseMatrixCSC(A)[I,J]
Base.copy(A::CuSparseMatrixBSR) = copyto!(similar(A),A)
Base.similar(A::CuSparseArrayCSR) = CuSparseArrayCSR(copy(A.rowPtr),copy(A.colVal),similar(A.nzVal),size(A))
Base.copy(A::CuSparseArrayCSR) =
    CuSparseArrayCSR(copy(A.rowPtr),copy(A.colVal),copy(A.nzVal),size(A))
function Base.show(io::IO,::MIME"text/plain",A::CuSparseArrayCSR)
    print(io,join(size(A),"×")," ",typeof(A)," with ",nnz(A)," stored ",
          nnz(A)==1 ? "entry" : "entries")
    all(>(0),size(A)) || return
    println(io,":")
    for (k,c) in enumerate(CartesianIndices(size(A)[3:end]))
        k>1 && println(io,"\n")
        println(io,"[:, :, ",join(c.I,", "),"] =")
        Base.print_array(IOContext(io,:typeinfo=>eltype(A)),SparseMatrixCSC(A[:,:,c.I...]))
    end
end
function Base.copyto!(dst::CuSparseArrayCSR,src::CuSparseArrayCSR)
    dst === src && return dst
    size(dst)==size(src) || throw(DimensionMismatch("batched sparse array dimensions must match"))
    all(size(d)==size(s) for (d,s) in zip((dst.rowPtr,dst.colVal,dst.nzVal),
                                        (src.rowPtr,src.colVal,src.nzVal))) ||
        throw(DimensionMismatch("batched sparse component sizes must match"))
    buffers(A)=(A.rowPtr,A.colVal,A.nzVal)
    any(d -> any(s -> Base.mightalias(d,s),buffers(src)),buffers(dst)) && (src=copy(src))
    GPUArrays.check_sparse_values(eltype(dst),vec(src.nzVal))
    GPUArrays.check_sparse_values(eltype(dst.rowPtr),vec(src.rowPtr))
    GPUArrays.check_sparse_values(eltype(dst.colVal),vec(src.colVal))
    p=GPUArrays.convert_index_buffer(eltype(dst.rowPtr),vec(src.rowPtr))
    i=GPUArrays.convert_index_buffer(eltype(dst.colVal),vec(src.colVal))
    v=GPUArrays.convert_buffer(eltype(dst),vec(src.nzVal))
    copyto!(dst.rowPtr,p)
    copyto!(dst.colVal,i)
    copyto!(dst.nzVal,v)
    return dst
end
CuSparseArrayCSR(p::CuArray, i::CuArray, v::CuArray{T}, dims::NTuple{N,Integer}) where {T,N} =
    CuSparseArrayCSR{T,eltype(i),N}(p,i,v,dims)
CuSparseArrayCSR{T}(p::CuArray,i::CuArray,v::CuArray,dims::NTuple{N,Integer}) where {T,N} =
    CuSparseArrayCSR{T,eltype(i),N}(p,i,v,dims)

CuSparseMatrixBSR{T}(p::CuVector,i::CuVector,v::CuVector,dims,b,dir,n) where {T} =
    CuSparseMatrixBSR{T,eltype(i)}(p,i,v,dims,b,dir,n)
CuSparseMatrixBSR(p::CuVector,i::CuVector,v::CuVector{T},b,dir,n,dims) where {T} =
    CuSparseMatrixBSR{T,eltype(i)}(p,i,v,dims,b,dir,n)
CuSparseMatrixBSR{T,Ti}(A::CuSparseMatrixBSR) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(GPUArrays.convert_index_buffer(Ti,A.rowPtr),GPUArrays.convert_index_buffer(Ti,A.colVal),
                             (GPUArrays.check_sparse_values(T,A.nzVal);GPUArrays.convert_buffer(T,A.nzVal)),
                             size(A),A.blockDim,A.dir,A.nnzb)
CuSparseMatrixBSR{T}(A::CuSparseMatrixBSR{<:Any,Ti}) where {T,Ti} = CuSparseMatrixBSR{T,Ti}(A)
CuSparseMatrixBSR{T,Ti}(A::SparseMatrixCSC,b) where {T,Ti} = CuSparseMatrixBSR(CuSparseMatrixCSR{T,Ti}(A),b)
CuSparseMatrixBSR{T}(A::SparseMatrixCSC,b) where {T} = CuSparseMatrixBSR{T,Cint}(A,b)
CuSparseMatrixBSR(A::SparseMatrixCSC,b) = CuSparseMatrixBSR{eltype(A),Cint}(A,b)
SparseArrays.SparseMatrixCSC(A::CuSparseMatrixBSR) = SparseMatrixCSC(CuSparseMatrixCSR(A))

Base.similar(Mat::CuSparseMatrixBSR{Tv, Ti}) where {Tv, Ti} =
    CuSparseMatrixBSR{Tv, Ti}(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), size(Mat), Mat.blockDim, Mat.dir, Mat.nnzb)

Base.similar(Mat::CuSparseMatrixBSR{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    CuSparseMatrixBSR{T, Ti}(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat), T), size(Mat), Mat.blockDim, Mat.dir, Mat.nnzb)

Base.similar(Mat::CuSparseMatrixBSR, ::Type{T}, ::Type{Ti}) where {T, Ti} =
    CuSparseMatrixBSR{T, Ti}(copyto!(similar(Mat.rowPtr, Ti), Mat.rowPtr), copyto!(similar(Mat.colVal, Ti), Mat.colVal), similar(nonzeros(Mat), T), size(Mat), Mat.blockDim, Mat.dir, Mat.nnzb)

Base.similar(Mat::CuSparseMatrixBSR{<:Any, Ti}, ::Type{T}, N::Int, M::Int) where {T, Ti} =
    CuSparseMatrixBSR{T, Ti}(fill!(CuVector{Ti}(undef, cld(N,Mat.blockDim)+1),one(Ti)), CuVector{Ti}(undef,0), CuVector{T}(undef,0), (N,M), Mat.blockDim, Mat.dir, 0)

Base.similar(Mat::CuSparseMatrixBSR, ::Type{T}, ::Type{Ti}, N::Int, M::Int) where {T, Ti} =
    CuSparseMatrixBSR{T, Ti}(fill!(CuVector{Ti}(undef, cld(N,Mat.blockDim)+1),one(Ti)), CuVector{Ti}(undef,0), CuVector{T}(undef,0), (N,M), Mat.blockDim, Mat.dir, 0)

Base.similar(Mat::CuSparseMatrixBSR{Tv}, N::Int, M::Int) where {Tv} = similar(Mat, Tv, N, M)

Base.similar(Mat::CuSparseMatrixBSR, T::Type, dims::Tuple{Int, Int}) = similar(Mat, T, dims...)

Base.similar(Mat::CuSparseMatrixBSR, T::Type, Ti::Type, dims::Tuple{Int, Int}) = similar(Mat, T, Ti, dims...)

Base.similar(Mat::CuSparseMatrixBSR, dims::Tuple{Int, Int}) = similar(Mat, dims...)

function Base.copyto!(dst::CuSparseMatrixBSR,src::CuSparseMatrixBSR)
    dst === src && return dst
    size(dst)==size(src) || throw(ArgumentError("Inconsistent Sparse Matrix size"))
    buffers(A)=(A.rowPtr,A.colVal,A.nzVal)
    any(d -> any(s -> Base.mightalias(d,s),buffers(src)),buffers(dst)) &&
        (src=copy(src))
    Ti=eltype(dst.rowPtr)
    GPUArrays.check_sparse_dims(Ti,(cld(size(src,1),src.blockDim),cld(size(src,2),src.blockDim),Int(src.blockDim)))
    GPUArrays.check_sparse_nnz(Ti,src.nnzb)
    GPUArrays.check_sparse_values(eltype(dst),src.nzVal)
    # Convert every input before any destination resize or write.
    p=GPUArrays.convert_index_buffer(Ti,src.rowPtr)
    i=GPUArrays.convert_index_buffer(eltype(dst.colVal),src.colVal)
    v=GPUArrays.convert_buffer(eltype(dst),src.nzVal)
    GPUArrays.replace_buffer!(dst.rowPtr,p)
    GPUArrays.replace_buffer!(dst.colVal,i)
    GPUArrays.replace_buffer!(dst.nzVal,v)
    dst.blockDim=src.blockDim
    dst.dir=src.dir
    dst.nnzb=src.nnzb
    return dst
end


function Base.getindex(A::CuSparseMatrixBSR{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    i0_block, i0_idx = fldmod1(i0, A.blockDim)
    i1_block, i1_idx = fldmod1(i1, A.blockDim)
    block_idx = A.dir == 'R' ? (i0_idx-1)*A.blockDim+i1_idx : (i1_idx-1)*A.blockDim+i0_idx
    c1 = Int(A.rowPtr[i0_block])
    c2 = Int(A.rowPtr[i0_block+1]-1)
    result = zero(T)
    for k in c1:c2
        if A.colVal[k] == i1_block
            value = nonzeros(A)[(k-1)*A.blockDim*A.blockDim+block_idx]
            result = T === Bool ? result | value : result + value
        end
    end
    return result
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


# Vendor-only device representations remain backend owned.
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
function Adapt.adapt_structure(to::CUDACore.KernelAdaptor,A::CuSparseMatrixBSR{T,Ti}) where {T,Ti}
    p,i,v = adapt(to,A.rowPtr),adapt(to,A.colVal),adapt(to,A.nzVal)
    CuSparseDeviceMatrixBSR{T,Ti,typeof(p),typeof(v)}(p,i,v,size(A),A.blockDim,A.dir,A.nnzb)
end
function Adapt.adapt_structure(to::CUDACore.KernelAdaptor,A::CuSparseArrayCSR{T,Ti,N}) where {T,Ti,N}
    p,i,v = adapt(to,A.rowPtr),adapt(to,A.colVal),adapt(to,A.nzVal)
    CuSparseDeviceArrayCSR{T,Ti,N,typeof(p),typeof(v)}(p,i,v,size(A),A.nnz)
end

# BSR transfers go through the canonical CSR converter; never iterate device scalars.
Base.Array(A::CuSparseMatrixBSR) = Array(CuSparseMatrixCSR(A))
Base.collect(A::CuSparseMatrixBSR) = Array(A)
CUDACore.CuArray(A::CuSparseMatrixBSR) = CuArray(CuSparseMatrixCSR(A))
CUDACore.CuArray{T}(A::CuSparseMatrixBSR) where {T} = CuArray{T}(CuSparseMatrixCSR(A))
Adapt.adapt_structure(::Type{Array}, A::CuSparseMatrixBSR) = SparseMatrixCSC(A)
