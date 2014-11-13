# CUDA related types

import Base: length, size, eltype

export    
    CuIn, CuOut, CuInOut

# TODO: use custom pointer type, not convertible to Ptr, yet usable as one


#
# Device pointer
#

typealias DevicePtr{T} Ptr{T}

# FIXME: normal constructor doesn't work?
function deviceptr{T}(p::Ptr{T})
    p::DevicePtr{T}
end

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
