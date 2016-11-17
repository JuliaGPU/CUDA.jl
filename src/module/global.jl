# Module-scope global variables

export
    CuGlobal, get, set


immutable CuGlobal{T}
    # TODO: typed pointer
    devptr::DevicePtr{Void}
    nbytes::Cssize_t

    function CuGlobal(mod::CuModule, name::String)
        ptr_ref = Ref{Ptr{Void}}()
        nbytes_ref = Ref{Cssize_t}()
        @apicall(:cuModuleGetGlobal, (Ptr{Ptr{Void}}, Ptr{Cssize_t}, CuModule_t, Ptr{Cchar}), 
                                     ptr_ref, nbytes_ref, mod, name)
        if nbytes_ref[] != sizeof(T)
            throw(ArgumentError("size of global '$name' does not match type parameter type $T"))
        end
        @assert nbytes_ref[] == sizeof(T)

        return new(DevicePtr{Void}(ptr_ref[], CuCurrentContext()), nbytes_ref[])
    end
end

Base.unsafe_convert(::Type{DevicePtr{Void}}, var::CuGlobal) = var.devptr

Base.:(==)(a::CuGlobal, b::CuGlobal) = a.handle == b.handle
Base.hash(var::CuGlobal, h::UInt) = hash(var.devptr, h)

Base.eltype{T}(::Type{CuGlobal{T}}) = T

function Base.get{T}(var::CuGlobal{T})
    val_ref = Ref{T}()
    @apicall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                            val_ref, var.devptr, var.nbytes)
    return val_ref[]
end

function set{T}(var::CuGlobal{T}, val::T)
    val_ref = Ref{T}(val)
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            var.devptr, val_ref, var.nbytes)
end
