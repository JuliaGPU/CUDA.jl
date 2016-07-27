# Module-scope global variables

import Base: eltype

export
    CuGlobal, get, set


immutable CuGlobal{T}
    ptr::DevicePtr{Void}
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
        new(unsafe_convert(DevicePtr{Void}, ptr_ref[]), nbytes_ref[])
    end
end

eltype{T}(x::Type{CuGlobal{T}}) = T

function get{T}(var::CuGlobal{T})
    val_ref = Ref{T}()
    @apicall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                            val_ref, var.ptr.inner, var.nbytes)
    return val_ref[]
end

function set{T}(var::CuGlobal{T}, val::T)
    val_ref = Ref{T}(val)
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            var.ptr.inner, val_ref, var.nbytes)
end
