# Module-related types and auxiliary functions

export
    CuModule, CuModuleFile

include(joinpath("module", "jit.jl"))


const CuModule_t = Ptr{Cvoid}

"""
    CuModule(data, options::Dict{CUjit_option,Any})
    CuModuleFile(path, options::Dict{CUjit_option,Any})

Create a CUDA module from a data, or a file containing data. The data may be PTX code, a
CUBIN, or a FATBIN.

The `options` is an optional dictionary of JIT options and their respective value.
"""
mutable struct CuModule
    handle::CuModule_t
    ctx::CuContext

    function CuModule(data, options::Dict{CUjit_option,Any}=Dict{CUjit_option,Any}())
        handle_ref = Ref{CuModule_t}()

        options[ERROR_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
        @static if CUDAapi.DEBUG
            options[INFO_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
            options[LOG_VERBOSE] = true
        end
        optionKeys, optionVals = encode(options)

        try
            @apicall(:cuModuleLoadDataEx,
                     (Ptr{CuModule_t}, Ptr{Cchar}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}),
                     handle_ref, data, length(optionKeys), optionKeys, optionVals)
        catch err
            (err == ERROR_NO_BINARY_FOR_GPU || err == ERROR_INVALID_IMAGE || err == ERROR_INVALID_PTX) || rethrow(err)
            options = decode(optionKeys, optionVals)
            rethrow(CuError(err.code, options[ERROR_LOG_BUFFER]))
        end

        @static if CUDAapi.DEBUG
            options = decode(optionKeys, optionVals)
            if isempty(options[INFO_LOG_BUFFER])
                @debug """JIT info log is empty"""
            else
                @debug """JIT info log:
                          $(options[INFO_LOG_BUFFER])"""
            end
        end

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx)
        finalizer(unsafe_unload!, obj)
        return obj
    end
end

function unsafe_unload!(mod::CuModule)
    if isvalid(mod.ctx)
        @trace("Finalizing CuModule object at $(Base.pointer_from_objref(mod)))")
        @apicall(:cuModuleUnload, (CuModule_t,), mod)
    else
        @trace("Skipping finalizer for CuModule object at $(Base.pointer_from_objref(mod))) because context is no longer valid")
    end
end

Base.unsafe_convert(::Type{CuModule_t}, mod::CuModule) = mod.handle

Base.:(==)(a::CuModule, b::CuModule) = a.handle == b.handle
Base.hash(mod::CuModule, h::UInt) = hash(mod.handle, h)

CuModuleFile(path) = CuModule(read(path, String))

include(joinpath("module", "function.jl"))
include(joinpath("module", "global.jl"))
include(joinpath("module", "linker.jl"))
