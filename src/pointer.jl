# Context-owned pointer

# Wrapper pointer type to keep track of the associated context, to avoid destroying
# the context while there's still outstanding references to its memory.
struct OwnedPtr{T}
    ptr::Ptr{T}
    ctx::CuContext

    OwnedPtr{T}(ptr::Ptr{T}, ctx::CuContext) where {T} = new{T}(ptr,ctx)
end


## getters

Base.pointer(p::OwnedPtr) = p.ptr

Base.isnull(p::OwnedPtr{T}) where {T} = (p.ptr == C_NULL)
Base.eltype(::Type{OwnedPtr{T}}) where {T} = T


## conversions

# between Ptr and OwnedPtr
## simple conversions disallowed
Base.convert(::Type{Ptr{T}}, p::OwnedPtr{T}) where {T} =
    throw(InexactError(:convert, Ptr{T}, p))
Base.convert(::Type{OwnedPtr{T}}, p::Ptr{T}) where {T} =
    throw(InexactError(:convert, OwnedPtr{T}, p))
## unsafe ones are allowed
Base.unsafe_convert(::Type{Ptr{T}}, p::OwnedPtr) where {T} = Ptr{T}(pointer(p))

# between pointers of different types
Base.convert(::Type{OwnedPtr{T}}, p::OwnedPtr) where {T} =
    OwnedPtr{T}(reinterpret(Ptr{T}, pointer(p)), p.ctx)


## limited pointer arithmetic & comparison

Base.:(==)(a::OwnedPtr, b::OwnedPtr) = a.ctx == b.ctx && pointer(a) == pointer(b)

Base.isless(x::OwnedPtr, y::OwnedPtr) = Base.isless(pointer(x), pointer(y))
Base.:(-)(x::OwnedPtr, y::OwnedPtr) = pointer(x) - pointer(y)

Base.:(+)(x::OwnedPtr{T}, y::Integer) where {T} = OwnedPtr{T}(pointer(x) + y, x.ctx)
Base.:(-)(x::OwnedPtr{T}, y::Integer) where {T} = OwnedPtr{T}(pointer(x) - y, x.ctx)
Base.:(+)(x::Integer, y::OwnedPtr) = y + x
