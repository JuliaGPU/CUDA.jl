# Module-scope global variables

# TODO: improve this interface:
# - should be more dict-like: get and setindex(::name), haskey(::name)
# - globals(::Type)?

export CuGlobal, get, set, CuGlobalArray


"""
    CuGlobal{T}(mod::CuModule, name::String)

Acquires a typed global variable handle from a named global in a module.
"""
struct CuGlobal{T}
    buf::Mem.DeviceBuffer

    function CuGlobal{T}(mod::CuModule, name::String) where T
        ptr_ref = Ref{CuPtr{Cvoid}}()
        nbytes_ref = Ref{Csize_t}()
        cuModuleGetGlobal_v2(ptr_ref, nbytes_ref, mod, name)
        if nbytes_ref[] != sizeof(T)
            throw(ArgumentError("size of global '$name' does not match type parameter type $T"))
        end
        buf = Mem.DeviceBuffer(ptr_ref[], nbytes_ref[])

        return new{T}(buf)
    end
end

Base.cconvert(::Type{CuPtr{Cvoid}}, var::CuGlobal) = var.buf

Base.:(==)(a::CuGlobal, b::CuGlobal) = a.handle == b.handle
Base.hash(var::CuGlobal, h::UInt) = hash(var.ptr, h)

"""
    eltype(var::CuGlobal)

Return the element type of a global variable object.
"""
Base.eltype(::Type{CuGlobal{T}}) where {T} = T

"""
    Base.getindex(var::CuGlobal)

Return the current value of a global variable.
"""
function Base.getindex(var::CuGlobal{T}) where T
    val_ref = Ref{T}()
    cuMemcpyDtoH_v2(val_ref, var, var.buf.bytesize)
    return val_ref[]
end
# TODO: import Base: get?

"""
    Base.setindex(var::CuGlobal{T}, val::T)

Set the value of a global variable to `val`
"""
function Base.setindex!(var::CuGlobal{T}, val::T) where T
    val_ref = Ref{T}(val)
    cuMemcpyHtoD_v2(var, val_ref, var.buf.bytesize)
end

"""
    CuGlobalArray{T}(mod::CuModule, name::String, len::Integer)

Acquires a global array variable handle from a named global in a module.
"""
struct CuGlobalArray{T} # TODO: the functionality provided by this struct can most likely be merged into CuGlobal{T}
    buf::Mem.DeviceBuffer

    function CuGlobalArray{T}(mod::CuModule, name::String, len::Integer) where T
        ptr_ref = Ref{CuPtr{Cvoid}}()
        nbytes_ref = Ref{Csize_t}()
        cuModuleGetGlobal_v2(ptr_ref, nbytes_ref, mod, name)
        if nbytes_ref[] != (sizeof(T) * len)
            throw(ArgumentError("size of global array '$name' ($(nbytes_ref[])) does not match given size (sizeof($T) * $length)"))
        end
        buf = Mem.DeviceBuffer(ptr_ref[], nbytes_ref[])

        return new{T}(buf)
    end
end

Base.eltype(::Type{CuGlobalArray{T}}) where {T} = T

Base.sizeof(global_array::CuGlobalArray{T}) where T = sizeof(global_array.buf)

function Base.copyto!(global_array::CuGlobalArray{T}, src::Array{T}) where T
    if sizeof(src) != sizeof(global_array)
        throw(DimensionMismatch("size of `src` ($(sizeof(src))) does not match global array ($(sizeof(global_array)))"))
    end
    cuMemcpyHtoD_v2(global_array.buf, src, sizeof(src))
end
