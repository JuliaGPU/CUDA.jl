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

        # XXX: cuModuleLoadData is sensitive to memory pressure, and can segfault when
        #      running close to OOM on CUDA 11.2 / driver 460 (NVIDIA bug #3284677).
        #
        #      this happens often when using the stream-ordered memory allocator, because
        #      the reserve of available memory we try to maintain is often not actually
        #      available, but cached by the allocator. by configuring the allocator with a
        #      release threshold, we have it actually free up that memory, but that requires
        #      synchronizing all streams to make sure pending frees are actually executed.
        device_synchronize()

        # FIXME: maybe all CUDA API calls need to run under retry_reclaim?
        #        that would require a redesign of the memory pool,
        #        so maybe do so when we replace it with CUDA 11.2's pool.
        res = GC.@preserve data retry_reclaim(isequal(ERROR_OUT_OF_MEMORY)) do
            unsafe_cuModuleLoadDataEx(handle_ref, pointer(data),
                                      length(optionKeys),
                                      optionKeys, optionVals)
        end
        if res == ERROR_NO_BINARY_FOR_GPU ||
           res == ERROR_INVALID_IMAGE ||
           res == ERROR_INVALID_PTX
            options = decode(optionKeys, optionVals)
            throw(CuError(res, unsafe_string(pointer(options[JIT_ERROR_LOG_BUFFER]))))
        elseif res != SUCCESS
            throw_api_error(res)
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
    @finalize_in_ctx mod.ctx cuModuleUnload(mod)
end

Base.unsafe_convert(::Type{CUmodule}, mod::CuModule) = mod.handle

Base.:(==)(a::CuModule, b::CuModule) = a.handle == b.handle
Base.hash(mod::CuModule, h::UInt) = hash(mod.handle, h)

CuModuleFile(path) = CuModule(read(path, String))

include(joinpath("module", "function.jl"))
include(joinpath("module", "global.jl"))
include(joinpath("module", "linker.jl"))
