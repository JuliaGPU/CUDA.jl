# Context-owned pointer

export
    OwnedPtr, CU_NULL

# Wrapper pointer type to keep track of the associated context, to avoid destroying
# the context while there's still outstanding references to its memory.
@compat immutable OwnedPtr{T}
    ptr::Ptr{T}
    ctx::CuContext

    (::Type{OwnedPtr{T}}){T}(ptr::Ptr{T}, ctx::CuContext) = new{T}(ptr,ctx)
end

function Base.:(==)(a::OwnedPtr, b::OwnedPtr)
    return a.ctx == b.ctx && a.ptr == b.ptr
end

const CU_NULL = OwnedPtr{Void}(C_NULL, CuContext(C_NULL))

# simple conversions between Ptr and OwnedPtr are disallowed
Base.convert{T}(::Type{Ptr{T}}, p::OwnedPtr{T}) = throw(InexactError())
Base.convert{T}(::Type{OwnedPtr{T}}, p::Ptr{T}) = throw(InexactError())

# unsafe ones are allowed
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::OwnedPtr{T}) = p.ptr

# conversion between pointers of different types
Base.convert{T}(::Type{OwnedPtr{T}}, p::OwnedPtr) =
    OwnedPtr{T}(reinterpret(Ptr{T}, p.ptr), p.ctx)

# some convenience methods (which are already defined for Ptr{T},
# but due to the disallowed conversions we can't use those)
Base.isnull{T}(p::OwnedPtr{T}) = (p.ptr == C_NULL)
Base.eltype{T}(::Type{OwnedPtr{T}}) = T
