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

    function CuModule(mod::String)
        module_ref = Ref{Ptr{Void}}()

        is_data = true
        try
          is_data = !ispath(mod)
        catch
          is_data = true
        end

        # FIXME: this is pretty messy
        if is_data

        options = Dict{CUjit_option,Any}()
            options[CU_JIT_ERROR_LOG_BUFFER] = Array(UInt8, 1024*1024)
            if DEBUG
                options[CU_JIT_GENERATE_LINE_INFO] = true
                options[CU_JIT_GENERATE_DEBUG_INFO] = true

                options[CU_JIT_INFO_LOG_BUFFER] = Array(UInt8, 1024*1024)
                options[CU_JIT_LOG_VERBOSE] = true
            end
            optionKeys, optionValues = encode(options)

            try
                @cucall(:cuModuleLoadDataEx,
                        (Ref{Ptr{Void}}, Ptr{Cchar}, Cuint, Ref{CUjit_option},Ref{Ptr{Void}}),
                        module_ref, mod, length(optionKeys), optionKeys, optionValues)
            catch err
                err == ERROR_NO_BINARY_FOR_GPU || rethrow(err)
                options = decode(optionKeys, optionValues)
                rethrow(CuError(err.code, options[CU_JIT_ERROR_LOG_BUFFER]))
            end

            options = decode(optionKeys, optionValues)
            if DEBUG
                debug("JIT info log:\n", options[CU_JIT_INFO_LOG_BUFFER])
            end
        else
            @cucall(:cuModuleLoad, (Ref{Ptr{Void}}, Ptr{Cchar}), module_ref, mod)
        end

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

    function CuFunction(md::CuModule, name::String)
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

    function CuGlobal(md::CuModule, name::String)
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
