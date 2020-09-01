import Base: view

using Base: ScalarIndex, ViewIndex, Slice, @boundscheck,
            to_indices, compute_offset1, unsafe_length, _maybe_reshape_parent, index_ndims


## traits and properties

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


## construction

@inline function view(A::CuArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    @boundscheck begin
        # Base's boundscheck accesses the indices, so make sure they reside on the CPU.
        # this is expensive, but it's a bounds check after all.
        J_cpu = map(j->adapt(Array, j), J)
        checkbounds(A, J_cpu...)
    end
    J_gpu = map(j->adapt(CuArray, j), J)
    unsafe_view(A, J_gpu, CuIndexStyle(I...))
end

# for contiguous views just return a new CuArray
@inline function unsafe_view(A, I, ::Contiguous)
    unsafe_contiguous_view(_maybe_reshape_parent(A, index_ndims(I...)), I, cuviewlength(I...))
end
@inline function unsafe_contiguous_view(A::CuArray{T}, I::NTuple{N,ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M}
    offset = compute_offset1(A, 1, I) * sizeof(T)
    CuArray{T,M}(pointer(A) + offset, dims, A)
end

# fallback to SubArray when the view is not contiguous
@inline function unsafe_view(A, I, ::NonContiguous)
    Base.unsafe_view(_maybe_reshape_parent(A, index_ndims(I...)), I...)
end


## operations

# upload the SubArray indices when adapting to the GPU
# (can't do this eagerly or the view constructor wouldn't be able to boundscheck)
Adapt.adapt_structure(to::Adaptor, A::SubArray) =
    SubArray(adapt(to, parent(A)), adapt(to, adapt(CuArray, parentindices(A))))
