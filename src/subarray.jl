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
    CuArray{T,M}(A.ptr + offset, dims, A)
end

# fallback to SubArray when the view is not contiguous
@inline _cuview(A, I, ::NonContiguous) where {N} =
    invoke(view, Tuple{AbstractArray, typeof(I).parameters...}, A, I...)


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
