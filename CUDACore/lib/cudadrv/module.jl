# Module-related types and auxiliary functions

export CuModule, CuModuleFile

include(joinpath("module", "jit.jl"))

function checked_cuModuleLoadDataEx(_module, image, numOptions, options, optionValues)
    # XXX: cuModuleLoadData is sensitive to memory pressure, and can segfault when
    #      running close to OOM on CUDA 11.2 / driver 460 (NVIDIA bug #3284677).
    #
    #      this happens often when using the stream-ordered memory allocator, because
    #      the reserve of available memory we try to maintain is often not actually
    #      available, but cached by the allocator. by configuring the allocator with a
    #      release threshold, we have it actually free up that memory, but that requires
    #      synchronizing all streams to make sure pending frees are actually executed.
    #
    #      that would invalidate graphs that are being captured, so skip it then. like
    #      reclaiming memory, this makes captures that start meanwhile fail, as waiting
    #      for the GPU could deadlock if a kernel depends on the capturing task.
    is_capturing() || releasing(device_synchronize; blocking=true)

    # FIXME: maybe all CUDA API calls need to run under retry_reclaim?
    #        that would require a redesign of the memory pool,
    #        so maybe do so when we replace it with CUDA 11.2's pool.
    res = retry_reclaim(isequal(ERROR_OUT_OF_MEMORY)) do
        res = unchecked_cuModuleLoadDataEx(_module, image, numOptions, options, optionValues)
        if res == ERROR_OUT_OF_MEMORY
            # OOM is self-explanatory, whether retried here or returned below.
            discard_driver_log()
        end
        res
    end
    if res != SUCCESS
        throw(CuError(res, driver_log()))
    end

    return
end


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

    # whether CUDA.jl unloads the module (see `unsafe_wrap`)
    const owned::Bool

    function CuModule(data, options::Dict{CUjit_option,Any}=Dict{CUjit_option,Any}())
        handle_ref = Ref{CUmodule}()

        options[JIT_ERROR_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
        if isdebug()
            options[JIT_INFO_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
            options[JIT_LOG_VERBOSE] = true
        end
        optionKeys, optionVals = encode(options)

        try
            GC.@preserve data begin
                checked_cuModuleLoadDataEx(handle_ref, pointer(data),
                                           length(optionKeys),
                                           optionKeys, optionVals)
            end
        catch err
            if isa(err, CuError) && err.code in (ERROR_NO_BINARY_FOR_GPU,
                                                 ERROR_INVALID_IMAGE,
                                                 ERROR_INVALID_PTX)
                options = decode(optionKeys, optionVals)
                log = GC.@preserve options unsafe_string(pointer(options[JIT_ERROR_LOG_BUFFER]))
                # the JIT log is only populated when compiling or linking; if the driver
                # rejected the image outright, the CuError (and its driver log) is all we have
                isempty(log) && rethrow()
                error(log)
            else
                rethrow()
            end
        end

        if isdebug()
            options = decode(optionKeys, optionVals)
            if !isempty(options[JIT_INFO_LOG_BUFFER])
                @debug """JIT info log:
                          $(options[JIT_INFO_LOG_BUFFER])""" _group=:CUDA
            end
        end

        ctx = current_context()
        obj = new(handle_ref[], ctx, true)
        # unloading a module waits for running kernels to finish
        resource_finalizer(unsafe_unload!, obj; ctx)
        return obj
    end

    global function Base.unsafe_wrap(::Type{CuModule}, handle::CUmodule; ctx::CuContext,
                                     own::Bool=false)
        obj = new(handle, ctx, own)
        own && resource_finalizer(unsafe_unload!, obj; ctx)
        return obj
    end
end

"""
    unsafe_wrap(CuModule, handle::CUmodule; ctx::CuContext, own=false)

Wrap a module that was loaded outside of CUDA.jl, in context `ctx`. With `own=true`, CUDA.jl
takes ownership of the module and unloads it when the returned object is garbage
collected; that is not possible for modules that belong to a library or to the CUDA
runtime. Otherwise, the handle needs to remain valid for as long as the returned object,
or a function obtained from it, is used.
"""
unsafe_wrap(::Type{CuModule}, ::CUmodule)

function unsafe_unload!(mod::CuModule)
    mod.owned || throw(ArgumentError("Cannot unload a module that CUDA.jl doesn't own"))
    context!(mod.ctx) do
        cuModuleUnload(mod)
    end
end

Base.unsafe_convert(::Type{CUmodule}, mod::CuModule) = mod.handle

Base.:(==)(a::CuModule, b::CuModule) = a.handle == b.handle
Base.hash(mod::CuModule, h::UInt) = hash(mod.handle, h)

CuModuleFile(path) = CuModule(read(path, String))

include(joinpath("module", "function.jl"))
include(joinpath("module", "global.jl"))
include(joinpath("module", "linker.jl"))
