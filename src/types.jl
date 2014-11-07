# CUDA related types


#
# device pointer
#

typealias CuPtr Ptr{Void}

CuPtr() = Ptr{Void}(0)

isnull(p::CuPtr) = (p == 0)


#
# managed data containers
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
