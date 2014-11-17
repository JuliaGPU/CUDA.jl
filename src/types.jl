# CUDA related types

import Base: length, size, eltype

export    
    CuIn, CuOut, CuInOut


#
# Device pointer
#

# TODO: a custom DevicePtr class would be nice, but Julia doesn't reason through
#       nested immutable classes (CuDeviceArray -> DevicePtr -> Ptr),
#       resulting in boxed accesses in CuDeviceArray.get/setindex
# immutable DevicePtr{T}
#     ptr::Ptr{T}
# end

typealias DevicePtr{T} Ptr{T}

isnull{T}(p::DevicePtr{T}) = (p == 0)


#
# Managed data containers
#

abstract CuManaged{T}

length(i::CuManaged) = length(i.data)
size(i::CuManaged) = size(i.data)
eltype{T}(i::CuManaged{T}) = T

type CuIn{T} <: CuManaged{T}
    data::T
end

type CuOut{T} <: CuManaged{T}
    data::T
end

type CuInOut{T} <: CuManaged{T}
    data::T
end

# FIXME: define on CuManaged instead of on each instance?
#eltype{T}(::Type{CuManaged{T}}) = T
eltype{T}(::Type{CuIn{T}}) = T
eltype{T}(::Type{CuOut{T}}) = T
eltype{T}(::Type{CuInOut{T}}) = T
eltype{T<:CuManaged}(t::Type{T}) =
    error("missing eltype definition for this managed type")
