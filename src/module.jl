# Module-related types and auxiliary functions

export
    CuModule, CuModuleFile, unload

include("module/jit.jl")


typealias CuModule_t Ptr{Void}

immutable CuModule
    handle::CuModule_t

    """
    Create a CUDA module from a string containing PTX code.

    If the Julia debug level is 2 or higher (or, on 0.5, if CUDAdrv is loaded in DEBUG
    mode), line number and debug information will be requested when loading the PTX code.
    """
    function CuModule(data)
        handle_ref = Ref{CuModule_t}()

        options = Dict{CUjit_option,Any}()
        options[ERROR_LOG_BUFFER] = Array(UInt8, 1024*1024)
        @static if VERSION >= v"0.6.0-dev.779"
            # TODO: put this in CUDAnative and document it
            if DEBUG || Base.JLOptions().debug_level >= 1
                options[GENERATE_LINE_INFO] = true
            end
            if DEBUG || Base.JLOptions().debug_level >= 2
                options[GENERATE_DEBUG_INFO] = true
            end
        end
        @static if DEBUG
            options[INFO_LOG_BUFFER] = Array(UInt8, 1024*1024)
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
            rethrow(CuError(err.code, options[ERROR_LOG_BUFFER]))
        end

        @static if DEBUG
            options = decode(optionKeys, optionVals)
            if isempty(options[INFO_LOG_BUFFER])
                debug("JIT info log is empty")
            else
                debug("JIT info log: ", repr_indented(options[INFO_LOG_BUFFER]; abbrev=false))
            end
        end

        new(handle_ref[])
    end
end

Base.unsafe_convert(::Type{CuModule_t}, mod::CuModule) = mod.handle
Base.:(==)(a::CuModule, b::CuModule) = a.handle == b.handle

"""
Unload a CUDA module.
"""
function unload(mod::CuModule)
    @apicall(:cuModuleUnload, (CuModule_t,), mod)
end

"""
Create a CUDA module from a file containing PTX code.

Note that for improved error reporting, this does not rely on the corresponding CUDA driver
call, but opens and reads the file from within Julia instead.
"""
CuModuleFile(path) = CuModule(open(readstring, path))

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

include("module/linker.jl")
include("module/function.jl")
include("module/global.jl")
