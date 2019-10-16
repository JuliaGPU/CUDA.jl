# CUDA pointer types

export CuPtr, NULL, PtrOrCuPtr


#
# CUDA pointer
#

"""
    CuPtr{T}

A memory address that refers to data of type `T` that is accessible from the GPU. A `CuPtr`
is ABI compatible with regular `Ptr` objects, e.g. it can be used to `ccall` a function that
expects a `Ptr` to GPU memory, but it prevents erroneous conversions between the two.
"""
CuPtr

if sizeof(Ptr{Cvoid}) == 8
    primitive type CuPtr{T} 64 end
else
    primitive type CuPtr{T} 32 end
end

# constructor
CuPtr{T}(x::Union{Int,UInt,CuPtr}) where {T} = Base.bitcast(CuPtr{T}, x)

const NULL = CuPtr{Cvoid}(0)


## getters

Base.eltype(::Type{<:CuPtr{T}}) where {T} = T


## conversions

# to and from integers
## pointer to integer
Base.convert(::Type{T}, x::CuPtr) where {T<:Integer} = T(UInt(x))
## integer to pointer
Base.convert(::Type{CuPtr{T}}, x::Union{Int,UInt}) where {T} = CuPtr{T}(x)
Int(x::CuPtr)  = Base.bitcast(Int, x)
UInt(x::CuPtr) = Base.bitcast(UInt, x)

# between regular and CUDA pointers
Base.convert(::Type{<:Ptr}, p::CuPtr) =
    throw(ArgumentError("cannot convert a GPU pointer to a CPU pointer"))

# between CUDA pointers
Base.convert(::Type{CuPtr{T}}, p::CuPtr) where {T} = Base.bitcast(CuPtr{T}, p)

# defer conversions to unsafe_convert
Base.cconvert(::Type{<:CuPtr}, x) = x

# fallback for unsafe_convert
Base.unsafe_convert(::Type{P}, x::CuPtr) where {P<:CuPtr} = convert(P, x)


## limited pointer arithmetic & comparison

isequal(x::CuPtr, y::CuPtr) = (x === y)
isless(x::CuPtr{T}, y::CuPtr{T}) where {T} = x < y

Base.:(==)(x::CuPtr, y::CuPtr) = UInt(x) == UInt(y)
Base.:(<)(x::CuPtr,  y::CuPtr) = UInt(x) < UInt(y)
Base.:(-)(x::CuPtr,  y::CuPtr) = UInt(x) - UInt(y)

Base.:(+)(x::CuPtr, y::Integer) = oftype(x, Base.add_ptr(UInt(x), (y % UInt) % UInt))
Base.:(-)(x::CuPtr, y::Integer) = oftype(x, Base.sub_ptr(UInt(x), (y % UInt) % UInt))
Base.:(+)(x::Integer, y::CuPtr) = y + x



#
# GPU or CPU pointer
#

"""
    PtrOrCuPtr{T}

A special pointer type, ABI-compatible with both `Ptr` and `CuPtr`, for use in `ccall`
expressions to convert values to either a GPU or a CPU type (in that order). This is
required for CUDA APIs which accept pointers that either point to host or device memory.
"""
PtrOrCuPtr


if sizeof(Ptr{Cvoid}) == 8
    primitive type PtrOrCuPtr{T} 64 end
else
    primitive type PtrOrCuPtr{T} 32 end
end

function Base.cconvert(::Type{PtrOrCuPtr{T}}, val) where {T}
    # `cconvert` is always implemented for both `Ptr` and `CuPtr`, so pick the first result
    # that has done an actual conversion

    gpu_val = Base.cconvert(CuPtr{T}, val)
    if gpu_val !== val
        return gpu_val
    end

    cpu_val = Base.cconvert(Ptr{T}, val)
    if cpu_val !== val
        return cpu_val
    end

    return val
end

function Base.unsafe_convert(::Type{PtrOrCuPtr{T}}, val) where {T}
    # FIXME: this is expensive; optimize using isapplicable?
    ptr = try
        Base.unsafe_convert(Ptr{T}, val)
    catch
        try
            Base.unsafe_convert(CuPtr{T}, val)
        catch
            throw(ArgumentError("cannot convert to either a CPU or GPU pointer"))
        end
    end
    return Base.bitcast(PtrOrCuPtr{T}, ptr)
end
