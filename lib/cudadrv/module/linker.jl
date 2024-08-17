# Linking of different PTX modules

export
    CuLink, CuLinkImage, add_data!, add_file!, complete


"""
    CuLink()

Creates a pending JIT linker invocation.
"""
mutable struct CuLink
    handle::CUlinkState
    ctx::CuContext

    options::Dict{CUjit_option,Any}
    optionKeys::Vector{CUjit_option}
    optionVals::Vector{Ptr{Cvoid}}

    function CuLink(options::Dict{CUjit_option,Any}=Dict{CUjit_option,Any}())
        handle_ref = Ref{CUlinkState}()

        options[JIT_ERROR_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
        @debug begin
            options[JIT_INFO_LOG_BUFFER] = Vector{UInt8}(undef, 1024*1024)
            options[JIT_LOG_VERBOSE] = true
            "JIT compiling code" # FIXME: remove this useless message
        end
        if Base.JLOptions().debug_level == 1
            # XXX: does not apply to the linker
            options[JIT_GENERATE_LINE_INFO] = true
        elseif Base.JLOptions().debug_level >= 2
            options[JIT_GENERATE_DEBUG_INFO] = true
        end
        optionKeys, optionVals = encode(options)

        cuLinkCreate_v2(length(optionKeys), optionKeys, optionVals, handle_ref)

        ctx = current_context()
        obj = new(handle_ref[], ctx, options, optionKeys, optionVals)
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

function unsafe_destroy!(link::CuLink)
    context!(link.ctx; skip_destroyed=true) do
        cuLinkDestroy(link)
    end
end

Base.unsafe_convert(::Type{CUlinkState}, link::CuLink) = link.handle

Base.:(==)(a::CuLink, b::CuLink) = a.handle == b.handle
Base.hash(link::CuLink, h::UInt) = hash(link.handle, h)

function Base.show(io::IO, link::CuLink)
    print(io, "CuLink(")
    @printf(io, "%p", link.handle)
    print(io, ", ", link.ctx, ")")
end

"""
    add_data!(link::CuLink, name::String, code::String)

Add PTX code to a pending link operation.
"""
function add_data!(link::CuLink, name::String, code::String)
    GC.@preserve code begin
        # cuLinkAddData takes a Ptr{Cvoid} instead of a Cstring, because it accepts both
        # source and binary, so do the conversion (ensuring no embedded NULLs) ourselves
        data = Base.unsafe_convert(Cstring, code)

        try
            cuLinkAddData_v2(link, JIT_INPUT_PTX, pointer(data), length(code), name, 0,
                             C_NULL, C_NULL)
        catch err
            if isa(err, CuError) && err.code in (ERROR_NO_BINARY_FOR_GPU,
                                                 ERROR_INVALID_IMAGE,
                                                 ERROR_INVALID_PTX)
                options = decode(link.optionKeys, link.optionVals)
                error(unsafe_string(pointer(link.options[JIT_ERROR_LOG_BUFFER])))
            else
                rethrow()
            end
        end
    end
end

"""
    add_data!(link::CuLink, name::String, data::Vector{UInt8})

Add object code to a pending link operation.
"""
function add_data!(link::CuLink, name::String, data::Vector{UInt8})
    try
        cuLinkAddData_v2(link, JIT_INPUT_OBJECT, data, length(data), name, 0, C_NULL, C_NULL)
    catch err
        if isa(err, CuError) && err.code in (ERROR_NO_BINARY_FOR_GPU,
                                             ERROR_INVALID_IMAGE,
                                             ERROR_INVALID_PTX)
            options = decode(link.optionKeys, link.optionVals)
            error(unsafe_string(pointer(link.options[JIT_ERROR_LOG_BUFFER])))
        else
            rethrow()
        end
    end
end

"""
    add_file!(link::CuLink, path::String, typ::CUjitInputType)

Add data from a file to a link operation. The argument `typ` indicates the type of the
contained data.
"""
function add_file!(link::CuLink, path::String, typ::CUjitInputType)
    cuLinkAddFile_v2(link, typ, path, 0, C_NULL, C_NULL)

    return nothing
end

"""
The result of a linking operation.

This object keeps its parent linker object alive, as destroying a linker destroys linked
images too.
"""
struct CuLinkImage
    data::Array{UInt8}
    link::CuLink
end


"""
    complete(link::CuLink)

Complete a pending linker invocation, returning an output image.
"""
function complete(link::CuLink)
    cubin_ref = Ref{Ptr{Cvoid}}()
    size_ref = Ref{Csize_t}()

    try
        cuLinkComplete(link, cubin_ref, size_ref)
    catch err
        if isa(err, CuError) && (err.code == ERROR_NO_BINARY_FOR_GPU || err.code == ERROR_INVALID_IMAGE)
            options = decode(link.optionKeys, link.optionVals)
            error(options[JIT_ERROR_LOG_BUFFER])
        else
            rethrow()
        end
    end

    if isdebug(:CuLink)
        options = decode(link.optionKeys, link.optionVals)
        if !isempty(options[JIT_INFO_LOG_BUFFER])
            @debug """JIT info log:
                      $(options[JIT_INFO_LOG_BUFFER])"""
        end
    end

    data = unsafe_wrap(Array, convert(Ptr{UInt8}, cubin_ref[]), size_ref[])
    return CuLinkImage(data, link)
end

"""
    CuModule(img::CuLinkImage, ...)

Create a CUDA module from a completed linking operation. Options from `CuModule` apply.
"""
CuModule(img::CuLinkImage, args...) = CuModule(img.data, args...)
CuModule(img::CuLinkImage, options::Dict{CUjit_option,Any}) = CuModule(img.data, options)
