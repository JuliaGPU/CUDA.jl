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
        module_box = ptrbox(Ptr{Void})
        is_data = true
        try
          is_data = !ispath(mod)
        catch
          is_data = true
        end
        # FIXME: this is pretty messy
        fname = is_data ? (:cuModuleLoadData) : (:cuModuleLoad)
        @cucall(fname, (Ptr{Ptr{Void}}, Ptr{Cchar}), module_box, mod)
        new(ptrunbox(module_box))
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
        function_box = ptrbox(Ptr{Void})
        @cucall(:cuModuleGetFunction, (Ptr{Ptr{Void}}, Ptr{Void}, Ptr{Cchar}),
                                      function_box, md.handle, name)
        new(ptrunbox(function_box))
    end
end


#
# Module-scope global variables
#

# TODO: parametric type given knowledge about device type?
immutable CuGlobal{T}
    pointer::DevicePtr{Void}
    nbytes::Cssize_t

    function CuGlobal(md::CuModule, name::ASCIIString)
        dptr_box = ptrbox(DevicePtr{Void})
        bytes_box = ptrbox(Cssize_t)
        @cucall(:cuModuleGetGlobal,
                (Ptr{DevicePtr{Void}}, Ptr{Cssize_t}, Ptr{Void}, Ptr{Cchar}), 
                dptr_box, bytes_box, md.handle, name)
        @assert ptrunbox(bytes_box) == sizeof(T)
        new(ptrunbox(dptr_box), ptrunbox(bytes_box))
    end
end

eltype{T}(::CuGlobal{T}) = T

function get{T}(var::CuGlobal{T})
    val_box = ptrbox(T)
    @cucall(:cuMemcpyDtoH, (Ptr{Void}, DevicePtr{Void}, Csize_t),
                           val_box, var.pointer, var.nbytes)
    return ptrunbox(val_box)
end

function set{T}(var::CuGlobal{T}, val::T)
    val_box = ptrbox(T, val)
    @cucall(:cuMemcpyHtoD, (DevicePtr{Void}, Ptr{Void}, Csize_t),
                           var.pointer, val_box, var.nbytes)
end
