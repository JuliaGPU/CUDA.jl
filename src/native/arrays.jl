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

getindex{T}(a::CuDeviceArray{T}, i0::Real) =
    unsafe_load(a.ptr, Base.to_index(i0))::T
setindex!{T}(a::CuDeviceArray{T}, x::T, i0::Real) =
    unsafe_store!(a.ptr, x, Base.to_index(i0))
