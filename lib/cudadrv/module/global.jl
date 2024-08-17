# Module-scope global variables

# TODO: improve this interface:
# - should be more dict-like: get and setindex(::name), haskey(::name)
# - globals(::Type)?

export CuGlobal


"""
    CuGlobal{T}(mod::CuModule, name::String)

Acquires a typed global variable handle from a named global in a module.
"""
struct CuGlobal{T}
    buf::DeviceMemory

    function CuGlobal{T}(mod::CuModule, name::String) where T
        ptr_ref = Ref{CuPtr{Cvoid}}()
        nbytes_ref = Ref{Csize_t}()
        cuModuleGetGlobal_v2(ptr_ref, nbytes_ref, mod, name)
        if nbytes_ref[] != sizeof(T)
            throw(ArgumentError("size of global '$name' does not match type parameter type $T"))
        end
        buf = DeviceMemory(device(), context(), ptr_ref[], nbytes_ref[], false)

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
function Base.getindex(var::CuGlobal{T}; async::Bool=false, stream::CuStream=stream()) where T
    val_ref = Ref{T}()
    if async
        cuMemcpyDtoHAsync_v2(val_ref, var, var.buf.bytesize, stream)
    else
        cuMemcpyDtoH_v2(val_ref, var, var.buf.bytesize)
    end
    return val_ref[]
end
# TODO: import Base: get?

"""
    Base.setindex(var::CuGlobal{T}, val::T)

Set the value of a global variable to `val`
"""
function Base.setindex!(var::CuGlobal{T}, val::T; async::Bool=false, stream::CuStream=stream()) where T
    val_ref = Ref{T}(val)
    if async
        cuMemcpyHtoDAsync_v2(var, val_ref, var.buf.bytesize, stream)
    else
        cuMemcpyHtoD_v2(var, val_ref, var.buf.bytesize)
    end
end
