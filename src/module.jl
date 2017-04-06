# Module-related types and auxiliary functions

export
    CuModule, CuModuleFile

include("module/jit.jl")


const CuModule_t = Ptr{Void}

type CuModule
    handle::CuModule_t
    ctx::CuContext

    """
    Create a CUDA module from a string containing PTX code.

    If the Julia debug level is 2 or higher (or, on 0.5, if CUDAdrv is loaded in DEBUG
    mode), line number and debug information will be requested when loading the PTX code.
    """
    function CuModule(data, options::Dict{CUjit_option,Any}=Dict{CUjit_option,Any}())
        handle_ref = Ref{CuModule_t}()

        options[ERROR_LOG_BUFFER] = Array{UInt8}(1024*1024)
        @static if DEBUG
            options[INFO_LOG_BUFFER] = Array{UInt8}(1024*1024)
            options[LOG_VERBOSE] = true
        end
        optionKeys, optionVals = encode(options)

        try
            @apicall(:cuModuleLoadDataEx,
                     (Ptr{CuModule_t}, Ptr{Cchar}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Void}}),
                     handle_ref, data, length(optionKeys), optionKeys, optionVals)
        catch err
            (err == ERROR_NO_BINARY_FOR_GPU || err == ERROR_INVALID_IMAGE) || rethrow(err)
            options = decode(optionKeys, optionVals)
            rethrow(CuError{code(err)}(options[ERROR_LOG_BUFFER]))
        end

        @static if DEBUG
            options = decode(optionKeys, optionVals)
            if isempty(options[INFO_LOG_BUFFER])
                @debug("JIT info log is empty")
            else
                @debug("JIT info log: ", repr_indented(options[INFO_LOG_BUFFER]; abbrev=false))
            end
        end

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx)
        finalizer(obj, unload!)
        return obj
    end
end

function unload!(mod::CuModule)
    if isvalid(mod.ctx)
        @trace("Finalizing CuModule at $(Base.pointer_from_objref(mod)))")
        @apicall(:cuModuleUnload, (CuModule_t,), mod)
    else
        @trace("Skipping finalizer for CuModule at $(Base.pointer_from_objref(mod))) because context is no longer valid")
    end
end

Base.unsafe_convert(::Type{CuModule_t}, mod::CuModule) = mod.handle

Base.:(==)(a::CuModule, b::CuModule) = a.handle == b.handle
Base.hash(mod::CuModule, h::UInt) = hash(mod.handle, h)

"""
Create a CUDA module from a file containing PTX code.

Note that for improved error reporting, this does not rely on the corresponding CUDA driver
call, but opens and reads the file from within Julia instead.
"""
CuModuleFile(path) = CuModule(open(readstring, path))

include("module/linker.jl")
include("module/function.jl")
include("module/global.jl")
