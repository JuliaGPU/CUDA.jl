# Module-related types and auxiliary functions

import Base: unsafe_convert

export
    CuModule, CuModuleFile, unload

include("module/jit.jl")


typealias CuModule_t Ptr{Void}

immutable CuModule
    handle::CuModule_t

    "Create a CUDA module from a string containing PTX code."
    function CuModule(data)
        handle_ref = Ref{CuModule_t}()

        options = Dict{CUjit_option,Any}()
        options[ERROR_LOG_BUFFER] = Array(UInt8, 1024*1024)
        if DEBUG
            options[GENERATE_LINE_INFO] = true
            options[GENERATE_DEBUG_INFO] = true

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
                debug("JIT info log: ", repr_indented(options[INFO_LOG_BUFFER]))
            end
        end

        new(handle_ref[])
    end
end

unsafe_convert(::Type{CuModule_t}, mod::CuModule) = mod.handle

"Unload a CUDA module."
function unload(mod::CuModule)
    @apicall(:cuModuleUnload, (CuModule_t,), mod)
end

"Create a CUDA module from a file containing PTX code. Note that for improved error reporting, this does not rely on the corresponding CUDA driver call, yet opens the file from within Julia."
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
