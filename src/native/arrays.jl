# Native arrays

import Base: getindex, setindex!

export
    CuDeviceArray


#
# Device array
#

# This is an alloc-free implementation of an array type

# TODO: can we implement AbstractArray to inherit its useful functionality?
typealias CuDeviceArray Ptr

# TODO: use an immutable type instead
#immutable CuDeviceArray{T}
#    ptr::Ptr{T}
#end

getindex{T}(a::CuDeviceArray{T}, i0::Real) =
    unsafe_load(a, Base.to_index(i0))::T
setindex!{T}(a::CuDeviceArray{T}, x::T, i0::Real) =
    unsafe_store!(a, x, Base.to_index(i0))
