# CUDA device pointer type

export
    DevicePtr, CU_NULL


# This wrapper type contains a pointer, but prevents many conversions from and to
# regular pointers. This ensures we don't mix host and device pointers.

immutable DevicePtr{T}
    ptr::Ptr{T}

    DevicePtr(::Ptr{T}) = throw(InexactError())
    DevicePtr(::Bool, ptr::Ptr{T}) = new(ptr)   # "hidden" constructor, don't use directly
end

CU_NULL = DevicePtr{Void}(true, C_NULL)

# simple conversions between Ptr and DevicePtr are disallowed
DevicePtr(::Ptr) = throw(InexactError())
Base.convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = throw(InexactError())
Base.convert{T}(::Type{DevicePtr{T}}, p::Ptr{T}) = throw(InexactError())

# unsafe ones are allowed
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::DevicePtr{T}) = p.ptr
Base.unsafe_convert{T}(::Type{DevicePtr{T}}, p::Ptr{T}) = DevicePtr{T}(true, p)

Base.cconvert{P<:DevicePtr}(::Type{P}, x) = x # defer conversions to DevicePtr to unsafe_convert

# conversion between pointers of different types
Base.convert{T}(::Type{DevicePtr{T}}, p::DevicePtr) =
    DevicePtr{T}(true, reinterpret(Ptr{T}, p.ptr))

# some convenience methods (which are already defined for Ptr{T},
# but due to the disallowed conversions we can't use those)
Base.isnull{T}(p::DevicePtr{T}) = (p.ptr == C_NULL)
Base.eltype{T}(::Type{DevicePtr{T}}) = T
