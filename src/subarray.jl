import Base: view

using Base: ScalarIndex, ViewIndex, Slice, @boundscheck,
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
@inline cuviewlength(::Real, I...) = cuviewlength(I...) # skip scalar
@inline cuviewlength(i1::AbstractUnitRange, I...) = (unsafe_length(i1), cuviewlength(I...)...)
@inline cuviewlength(i1::AbstractUnitRange, ::ScalarIndex...) = (unsafe_length(i1),)

@inline view(A::CuArray, I::Vararg{Any,N}) where {N} = _cuview(A, I, CuIndexStyle(I...))

@inline function _cuview(A, I, ::Contiguous)
    J = to_indices(A, I)
    @boundscheck checkbounds(A, J...)
    _cuview(_maybe_reshape_parent(A, index_ndims(J...)), J, cuviewlength(J...))
end

# for contiguous views just return a new CuArray
@inline function _cuview(A::CuArray{T}, I::NTuple{N,ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M}
    offset = compute_offset1(A, 1, I) * sizeof(T)
    CuArray{T,M}(pointer(A) + offset, dims, A)
end

# fallback to SubArray when the view is not contiguous
@inline _cuview(A, I, ::NonContiguous) where {N} =
    invoke(view, Tuple{AbstractArray, typeof(I).parameters...}, A, I...)


## operations

# upload the SubArray indices when adapting to the GPU
# (can't do this eagerly or the view constructor wouldn't be able to boundscheck)
# FIXME: alternatively, have users do `cu(view(cu(A), inds))`, but that seems redundant
Adapt.adapt_structure(to::Adaptor, A::SubArray) =
    SubArray(adapt(to, parent(A)), adapt(to, adapt(CuArray, parentindices(A))))
