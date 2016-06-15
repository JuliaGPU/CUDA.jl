# Module-related types and auxiliary functions

import Base: eltype, unsafe_convert

export
    CuModule, CuModuleFile, CuModuleData, unload,
    CuFunction,
    CuGlobal, get, set


#
# CUDA module
#

typealias CuModule_t Ptr{Void}

abstract CuModule

immutable CuModuleFile <: CuModule
    handle::CuModule_t

    "Create a CUDA module from a file containing PTX code."
    function CuModuleFile(path)
        handle_ref = Ref{CuModule_t}()

        @apicall(:cuModuleLoad, (Ptr{CuModule_t}, Ptr{Cchar}), handle_ref, path)

        new(handle_ref[])
    end
end

# do syntax, f(module)
function CuModuleFile(f::Function, path::AbstractString)
    mod = CuModuleFile(path)
    local ret
    try
        ret = f(mod)
    finally
        unload(mod)
    end
    ret
end

immutable CuModuleData <: CuModule
    handle::CuModule_t

    "Create a CUDA module from a string containing PTX code."
    function CuModuleData(data)
        handle_ref = Ref{CuModule_t}()

        options = Dict{CUjit_option,Any}()
        options[ERROR_LOG_BUFFER] = Array(UInt8, 1024*1024)
        if DEBUG
            options[GENERATE_LINE_INFO] = true
            options[GENERATE_DEBUG_INFO] = true

            options[INFO_LOG_BUFFER] = Array(UInt8, 1024*1024)
            options[LOG_VERBOSE] = true
        end
        optionKeys, optionValues = encode(options)

        try
            @apicall(:cuModuleLoadDataEx,
                    (Ptr{CuModule_t}, Ptr{Cchar}, Cuint, Ref{CUjit_option}, Ref{Ptr{Void}}),
                    handle_ref, data, length(optionKeys), optionKeys, optionValues)
        catch err
            (err == ERROR_NO_BINARY_FOR_GPU || err == ERROR_INVALID_IMAGE) || rethrow(err)
            options = decode(optionKeys, optionValues)
            rethrow(CuError(err.code, options[ERROR_LOG_BUFFER]))
        end

        if DEBUG
            options = decode(optionKeys, optionValues)
            if isempty(options[INFO_LOG_BUFFER])
                debug("JIT info log is empty")
            else
                debug("JIT info log: ", repr_indented(options[INFO_LOG_BUFFER]))
            end
        end

        new(handle_ref[])
    end
end

unsafe_convert(::Type{CuModule_t}, mod::CuModule) = mod.handle

"Unload a CUDA module."
function unload(mod::CuModule)
    @apicall(:cuModuleUnload, (CuModule_t,), mod.handle)
end


#
# CUDA function
#

typealias CuFunction_t Ptr{Void}

immutable CuFunction
    handle::CuFunction_t

    "Get a handle to a kernel function in a CUDA module."
    function CuFunction(mod::CuModule, name::String)
        handle_ref = Ref{CuFunction_t}()
        @apicall(:cuModuleGetFunction, (Ptr{CuFunction_t}, CuModule_t, Ptr{Cchar}),
                                      handle_ref, mod.handle, name)
        new(handle_ref[])
    end
end

unsafe_convert(::Type{CuFunction_t}, fun::CuFunction) = fun.handle


#
# Module-scope global variables
#

# TODO: parametric type given knowledge about device type?
immutable CuGlobal{T}
    ptr::DevicePtr{Void}
    nbytes::Cssize_t

    function CuGlobal(mod::CuModule, name::String)
        ptr_ref = Ref{Ptr{Void}}()
        nbytes_ref = Ref{Cssize_t}()
        @apicall(:cuModuleGetGlobal,
                (Ptr{Ptr{Void}}, Ptr{Cssize_t}, Ptr{Void}, Ptr{Cchar}), 
                ptr_ref, nbytes_ref, mod.handle, name)
        if nbytes_ref[] != sizeof(T)
            throw(ArgumentError("type of global does not match type parameter type"))
        end
        @assert nbytes_ref[] == sizeof(T)
        new(DevicePtr{Void}(ptr_ref[], true), nbytes_ref[])
    end
end

eltype{T}(::CuGlobal{T}) = T

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
