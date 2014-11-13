# Native arrays

export
    UnsafeArray


immutable UnsafeArray{T} <: AbstractArray
    ptr::Ptr{T}
end

getindex{T}(A::UnsafeArray{T}, i0::Real) =
    unsafe_load(A.ptr, Base.to_index(i0))::T
setindex!{T}(A::UnsafeArray{T}, x::T, i0::Real) =
    unsafe_store!(A.ptr, x, Base.to_index(i0))
