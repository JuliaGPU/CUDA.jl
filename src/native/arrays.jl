# Native arrays

export
    CuDeviceArray


#
# Device array
#

# This is an alloc-free implementation of an array type

immutable CuDeviceArray{T} <: AbstractArray
    ptr::DevicePtr{T}
end

getindex{T}(A::CuDeviceArray{T}, i0::Real) =
    unsafe_load(A.ptr, Base.to_index(i0))::T
setindex!{T}(A::CuDeviceArray{T}, x::T, i0::Real) =
    unsafe_store!(A.ptr, x, Base.to_index(i0))
