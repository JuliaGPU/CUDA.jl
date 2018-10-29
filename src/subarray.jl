import Base: view

using Base: ScalarIndex, ViewIndex, Slice, @_inline_meta, @boundscheck, 
            to_indices, compute_offset1, unsafe_length

struct Contiguous end
struct NonContiguous end

# Detect whether the view is contiguous or not
CuIndexStyle() = Contiguous()
CuIndexStyle(i1::Colon, I...) = CuIndexStyle(I...)
CuIndexStyle(i1::AbstractUnitRange, ::ScalarIndex...) = Contiguous()
CuIndexStyle(I...) = NonContiguous()

cuviewlength() = ()
cuviewlength(::Real, I...) = (@_inline_meta; cuviewlength(I...)) # skip scalars
cuviewlength(i1::AbstractUnitRange, I...) = (@_inline_meta; (unsafe_length(i1), cuviewlength(I...)...))
cuviewlength(i1::AbstractUnitRange, ::ScalarIndex...) = (@_inline_meta; (unsafe_length(i1),))

view(A::CuArray, I::Vararg{Any,N}) where {N} = (@_inline_meta; _cuview(A, I, CuIndexStyle(I...)))

function _cuview(A, I, ::Contiguous)
    @_inline_meta
    J = to_indices(A, I)
    @boundscheck checkbounds(A, J...)
    _cuview(A, J, cuviewlength(J...))
end

# for contiguous views just return a new CuArray
_cuview(A::CuArray{T}, I::NTuple{N,ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M} =
    CuArray{T,M}(A.buf, A.offset + compute_offset1(A, 1, I) * sizeof(T), dims)

# fallback to SubArray when the view is not contiguous
_cuview(A, I::NTuple{N,ViewIndex}, ::NonContiguous) where {N} = invoke(view, Tuple{AbstractArray, typeof(I).parameters...}, A, I...)