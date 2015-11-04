# Module-related types and auxiliary functions

import Base: eltype

export
    CuModule, unload,
    CuFunction,
    CuGlobal, get, set


#
# CUDA module
#

immutable CuModule
    handle::Ptr{Void}

    function CuModule(mod::ASCIIString)
        module_ref = Ref{Ptr{Void}}()

        is_data = true
        try
          is_data = !ispath(mod)
        catch
          is_data = true
        end
        # FIXME: this is pretty messy
        fname = is_data ? (:cuModuleLoadData) : (:cuModuleLoad)

        @cucall(fname, (Ptr{Ptr{Void}}, Ptr{Cchar}), module_ref, mod)
        new(module_ref[])
    end
end

function unload(md::CuModule)
    @cucall(:cuModuleUnload, (Ptr{Void},), md.handle)
end


#
# CUDA function
#

immutable CuFunction
    handle::Ptr{Void}

    function CuFunction(md::CuModule, name::ASCIIString)
        function_ref = Ref{Ptr{Void}}()
        @cucall(:cuModuleGetFunction, (Ptr{Ptr{Void}}, Ptr{Void}, Ptr{Cchar}),
                                      function_ref, md.handle, name)
        new(function_ref[])
    end
end


#
# Module-scope global variables
#

# TODO: parametric type given knowledge about device type?
immutable CuGlobal{T}
    ptr::DevicePtr{Void}
    nbytes::Cssize_t

    function CuGlobal(md::CuModule, name::ASCIIString)
        ptr_ref = Ref{Ptr{Void}}()
        bytes_ref = Ref{Cssize_t}()
        @cucall(:cuModuleGetGlobal,
                (Ptr{Ptr{Void}}, Ptr{Cssize_t}, Ptr{Void}, Ptr{Cchar}), 
                ptr_ref, bytes_ref, md.handle, name)
        @assert bytes_ref[] == sizeof(T)
        new(DevicePtr{Void}(ptr_ref[], true), bytes_ref[])
    end
end

eltype{T}(::CuGlobal{T}) = T

function get{T}(var::CuGlobal{T})
    val_ref = Ref{T}()
    @cucall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                           val_ref, var.ptr.inner, var.nbytes)
    return val_ref[]
end

function set{T}(var::CuGlobal{T}, val::T)
    val_ref = Ref{T}(val)
    @cucall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                           var.ptr.inner, val_ref, var.nbytes)
end
