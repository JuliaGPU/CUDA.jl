# Context-owned pointer

# Wrapper pointer type to keep track of the associated context, to avoid destroying
# the context while there's still outstanding references to its memory.
immutable OwnedPtr{T}
    ptr::Ptr{T}
    ctx::CuContext

    (::Type{OwnedPtr{T}}){T}(ptr::Ptr{T}, ctx::CuContext) = new{T}(ptr,ctx)
end


## getters

Base.pointer(p::OwnedPtr) = p.ptr

Base.isnull{T}(p::OwnedPtr{T}) = (p.ptr == C_NULL)
Base.eltype{T}(::Type{OwnedPtr{T}}) = T


## conversions

# between Ptr and OwnedPtr
## simple conversions disallowed
Base.convert{T}(::Type{Ptr{T}}, p::OwnedPtr{T}) = throw(InexactError())
Base.convert{T}(::Type{OwnedPtr{T}}, p::Ptr{T}) = throw(InexactError())
## unsafe ones are allowed
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::OwnedPtr) = Ptr{T}(pointer(p))

# between pointers of different types
Base.convert{T}(::Type{OwnedPtr{T}}, p::OwnedPtr) =
    OwnedPtr{T}(reinterpret(Ptr{T}, pointer(p)), p.ctx)


## limited pointer arithmetic & comparison

Base.:(==)(a::OwnedPtr, b::OwnedPtr) = a.ctx == b.ctx && pointer(a) == pointer(b)

Base.isless(x::OwnedPtr, y::OwnedPtr) = Base.isless(pointer(x), pointer(y))
Base.:(-)(x::OwnedPtr, y::OwnedPtr) = pointer(x) - pointer(y)

Base.:(+){T}(x::OwnedPtr{T}, y::Integer) = OwnedPtr{T}(pointer(x) + y, x.ctx)
Base.:(-){T}(x::OwnedPtr{T}, y::Integer) = OwnedPtr{T}(pointer(x) - y, x.ctx)
Base.:(+)(x::Integer, y::OwnedPtr) = y + x
