import Base: view

using Base: ScalarIndex, ViewIndex, Slice, @_inline_meta, @boundscheck,
            to_indices, compute_offset1, unsafe_length, _maybe_reshape_parent, index_ndims


## construction

struct Contiguous end
struct NonContiguous end

# Detect whether the view is contiguous or not
CuIndexStyle() = Contiguous()
CuIndexStyle(I...) = NonContiguous()
CuIndexStyle(i1::Colon, ::ScalarIndex...) = Contiguous()
CuIndexStyle(i1::AbstractUnitRange, ::ScalarIndex...) = Contiguous()
CuIndexStyle(i1::Colon, I...) = CuIndexStyle(I...)

cuviewlength() = ()
cuviewlength(::Real, I...) = (@_inline_meta; cuviewlength(I...)) # skip scalars
cuviewlength(i1::AbstractUnitRange, I...) = (@_inline_meta; (unsafe_length(i1), cuviewlength(I...)...))
cuviewlength(i1::AbstractUnitRange, ::ScalarIndex...) = (@_inline_meta; (unsafe_length(i1),))

view(A::CuArray, I::Vararg{Any,N}) where {N} = (@_inline_meta; _cuview(A, I, CuIndexStyle(I...)))

function _cuview(A, I, ::Contiguous)
    @_inline_meta
    J = to_indices(A, I)
    @boundscheck checkbounds(A, J...)
    _cuview(_maybe_reshape_parent(A, index_ndims(J...)), J, cuviewlength(J...))
end

# for contiguous views just return a new CuArray
function _cuview(A::CuArray{T}, I::NTuple{N,ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M}
    offset = compute_offset1(A, 1, I) * sizeof(T)
    CuArray{T,M}(A.ptr + offset, dims; base=A.base, own=A.own)
end

# fallback to SubArray when the view is not contiguous
_cuview(A, I, ::NonContiguous) where {N} = invoke(view, Tuple{AbstractArray, typeof(I).parameters...}, A, I...)


## operations

# copyto! doesn't know how to deal with SubArrays, but broadcast does
# FIXME: use the rules from Adapt.jl to define copyto! methods in GPUArrays.jl
function Base.copyto!(dest::GPUArray{T,N}, src::SubArray{T,N,<:GPUArray{T}}) where {T,N}
    view(dest, axes(src)...) .= src
    dest
end

# copying to a CPU array requires an intermediate copy
# TODO: support other copyto! invocations (GPUArrays.jl copyto! defs + Adapt.jl rules)
function Base.copyto!(dest::AbstractArray{T,N}, src::SubArray{T,N,AT}) where {T,N,AT<:GPUArray{T}}
    temp = similar(AT, axes(src))
    copyto!(temp, src)
    copyto!(dest, temp)
end

# upload the SubArray indices when adapting to the GPU
# (can't do this eagerly or the view constructor wouldn't be able to boundscheck)
# FIXME: alternatively, have users do `cu(view(cu(A), inds))`, but that seems redundant
Adapt.adapt_structure(to::CUDAnative.Adaptor, A::SubArray) =
    SubArray(adapt(to, parent(A)), adapt(to, adapt(CuArray, parentindices(A))))
