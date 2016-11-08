# CUDA device pointer type

export
    DevicePtr, CU_NULL


# This wrapper type contains a pointer, but prevents many conversions from and to
# regular pointers. This ensures we don't mix host and device pointers.
#
# It also keep track of the associated context, preventing it from getting freed while
# there's still a pointer from that context live.

immutable DevicePtr{T}
    ptr::Ptr{T}
    ctx::CuContext

    DevicePtr(ptr::Ptr{T}, ctx::CuContext) = new(ptr,ctx)
end

function Base.:(==)(a::DevicePtr, b::DevicePtr)
    return a.ctx == b.ctx && a.ptr == b.ptr
end

CU_NULL = DevicePtr{Void}(C_NULL, CuContext(C_NULL))

# simple conversions between Ptr and DevicePtr are disallowed
Base.convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = throw(InexactError())
Base.convert{T}(::Type{DevicePtr{T}}, p::Ptr{T}) = throw(InexactError())

# unsafe ones are allowed
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = p.ptr

Base.cconvert{P<:DevicePtr}(::Type{P}, x) = x # defer conversions to DevicePtr to unsafe_convert

# conversion between pointers of different types
Base.convert{T}(::Type{DevicePtr{T}}, p::DevicePtr) =
    DevicePtr{T}(reinterpret(Ptr{T}, p.ptr), p.ctx)

# some convenience methods (which are already defined for Ptr{T},
# but due to the disallowed conversions we can't use those)
Base.isnull{T}(p::DevicePtr{T}) = (p.ptr == C_NULL)
Base.eltype{T}(::Type{DevicePtr{T}}) = T
