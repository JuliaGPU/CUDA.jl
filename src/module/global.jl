# Module-scope global variables

export
    CuGlobal, get, set


"""
    CuGlobal{T}(mod::CuModule, name::String)

Acquires a typed global variable handle from a named global in a module.
"""
@compat immutable CuGlobal{T}
    # TODO: typed pointer
    ptr::OwnedPtr{Void}
    nbytes::Cssize_t

    function (::Type{CuGlobal{T}}){T}(mod::CuModule, name::String)
        ptr_ref = Ref{Ptr{Void}}()
        nbytes_ref = Ref{Cssize_t}()
        @apicall(:cuModuleGetGlobal, (Ptr{Ptr{Void}}, Ptr{Cssize_t}, CuModule_t, Ptr{Cchar}), 
                                     ptr_ref, nbytes_ref, mod, name)
        if nbytes_ref[] != sizeof(T)
            throw(ArgumentError("size of global '$name' does not match type parameter type $T"))
        end
        @assert nbytes_ref[] == sizeof(T)

        return new{T}(OwnedPtr{Void}(ptr_ref[], CuCurrentContext()), nbytes_ref[])
    end
end

Base.unsafe_convert(::Type{OwnedPtr{Void}}, var::CuGlobal) = var.ptr

Base.:(==)(a::CuGlobal, b::CuGlobal) = a.handle == b.handle
Base.hash(var::CuGlobal, h::UInt) = hash(var.ptr, h)

"""
    eltype(var::CuGlobal)

Return the element type of a global variable object.
"""
Base.eltype{T}(::Type{CuGlobal{T}}) = T

"""
    get(var::CuGlobal)

Return the current value of a global variable.
"""
function Base.get{T}(var::CuGlobal{T})
    val_ref = Ref{T}()
    @apicall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                            val_ref, var.ptr, var.nbytes)
    return val_ref[]
end

"""
    set(var::CuGlobal{T}, T)

Set the value of a global variable to `val`
"""
function set{T}(var::CuGlobal{T}, val::T)
    val_ref = Ref{T}(val)
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            var.ptr, val_ref, var.nbytes)
end
