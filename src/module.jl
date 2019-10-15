# Module-related types and auxiliary functions

export
    CuModule, CuModuleFile

include(joinpath("module", "jit.jl"))


"""
    CuModule(data, options::Dict{CUjit_option,Any})
    CuModuleFile(path, options::Dict{CUjit_option,Any})

Create a CUDA module from a data, or a file containing data. The data may be PTX code, a
CUBIN, or a FATBIN.

The `options` is an optional dictionary of JIT options and their respective value.
"""
mutable struct CuModule
    handle::CUmodule
    ctx::CuContext

    function CuModule(data, options::Dict{CUjit_option,Any}=Dict{CUjit_option,Any}())
        handle_ref = Ref{CUmodule}()

        options[JIT_ERROR_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
        @debug begin
            options[JIT_INFO_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
            options[JIT_LOG_VERBOSE] = true
            "JIT compiling code" # FIXME: remove this useless message
        end
        optionKeys, optionVals = encode(options)

        err = @apicall_nothrow(:cuModuleLoadDataEx,
                               (Ptr{CUmodule}, Ptr{Cchar}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}),
                               handle_ref, data, length(optionKeys), optionKeys, optionVals)
        if err == ERROR_NO_BINARY_FOR_GPU || err == ERROR_INVALID_IMAGE || err == ERROR_INVALID_PTX
            options = decode(optionKeys, optionVals)
            throw(CuError(err.code, options[JIT_ERROR_LOG_BUFFER]))
        elseif err != SUCCESS
            throw(err)
        end

        @debug begin
            options = decode(optionKeys, optionVals)
            if isempty(options[JIT_INFO_LOG_BUFFER])
                """JIT info log is empty"""
            else
                """JIT info log:
                   $(options[JIT_INFO_LOG_BUFFER])"""
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
        @apicall(:cuModuleUnload, (CUmodule,), mod)
    end
end

Base.unsafe_convert(::Type{CUmodule}, mod::CuModule) = mod.handle

Base.:(==)(a::CuModule, b::CuModule) = a.handle == b.handle
Base.hash(mod::CuModule, h::UInt) = hash(mod.handle, h)

CuModuleFile(path) = CuModule(read(path, String))

include(joinpath("module", "function.jl"))
include(joinpath("module", "global.jl"))
include(joinpath("module", "linker.jl"))
