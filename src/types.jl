# CUDA related types

import Base: eltype, convert, isnull, cconvert


#
# Device pointer
#

# This wrapper type contains a pointer, but prevents all conversions from and to
# regular pointers. This ensures we don't mix host and device pointers.

# NOTE: this type is only for use in host code,
#       device code is currently still generated using plain Ptr's

import Base: getindex

immutable DevicePtr{T}
    inner::Ptr{T}

    DevicePtr() = new(C_NULL)
    DevicePtr{T}(::Ptr{T}) = throw(InexactError())
    DevicePtr{T}(x::Ptr{T}, really::Bool) = new(x)
    # NOTE: the reasoning behind this strange (Ptr, Bool) constructor, which is
    # the only possible way of setting the inner pointer, is to make sure the
    # user knows what he's doing: constructing a device pointer from a regular
    # pointer value
end

# Simple conversions between Ptr and DevicePtr are disallowed
convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = throw(InexactError())
convert{T}(::Type{DevicePtr{T}}, p::Ptr{T}) = throw(InexactError())

# Some convenience methods (which are already defined for Ptr{T},
# but due to the disallowed conversions we can't use those)
isnull{T}(p::DevicePtr{T}) = (p.inner == 0)
eltype{T}(x::Type{DevicePtr{T}}) = T
