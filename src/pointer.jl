# CUDA device pointer type

import Base: eltype, convert, unsafe_convert, cconvert, isnull

export
    DevicePtr


# This wrapper type contains a pointer, but prevents all conversions from and to
# regular pointers. This ensures we don't mix host and device pointers.

immutable DevicePtr{T}
    inner::Ptr{T}

    DevicePtr() = new(C_NULL)
    DevicePtr{T}(::Ptr{T}) = throw(InexactError())
    DevicePtr{T}(::Bool, x::Ptr{T}) = new(x)
    # NOTE: this hidden constructor is not meant to be used -- use unsafe_convert instead
end

# Simple conversions between Ptr and DevicePtr are disallowed
convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = throw(InexactError())
convert{T}(::Type{DevicePtr{T}}, p::Ptr{T}) = throw(InexactError())

# Unsafe ones are tolerated
unsafe_convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = p.inner
unsafe_convert{T}(::Type{DevicePtr{T}}, p::Ptr{T}) = DevicePtr{T}(true, p)

cconvert{P<:DevicePtr}(::Type{P}, x) = x # defer conversions to DevicePtr to unsafe_convert

# Some convenience methods (which are already defined for Ptr{T},
# but due to the disallowed conversions we can't use those)
# TODO: verify these are banned
isnull{T}(p::DevicePtr{T}) = (p.inner == C_NULL)
eltype{T}(::Type{DevicePtr{T}}) = T
