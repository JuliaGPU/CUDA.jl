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
            options[JIT_GENERATE_LINE_INFO] = true
        elseif Base.JLOptions().debug_level >= 2
            options[JIT_GENERATE_DEBUG_INFO] = true
        end
        optionKeys, optionVals = encode(options)

        cuLinkCreate(length(optionKeys), optionKeys, optionVals, handle_ref)

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx, options, optionKeys, optionVals)
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

function unsafe_destroy!(link::CuLink)
    if isvalid(link.ctx)
        cuLinkDestroy(link)
    end
end

Base.unsafe_convert(::Type{CUlinkState}, link::CuLink) = link.handle

Base.:(==)(a::CuLink, b::CuLink) = a.handle == b.handle
Base.hash(link::CuLink, h::UInt) = hash(link.handle, h)

"""
    add_data!(link::CuLink, name::String, code::String)

Add PTX code to a pending link operation.
"""
function add_data!(link::CuLink, name::String, code::String)
    data = unsafe_wrap(Vector{UInt8}, code)

    # there shouldn't be any embedded NULLs
    checked_data = Base.unsafe_convert(Cstring, data)

    cuLinkAddData(link, JIT_INPUT_PTX, pointer(checked_data), length(data), name, 0, C_NULL, C_NULL)
end

"""
    add_data!(link::CuLink, name::String, data::Vector{UInt8}, type::CUjitInputType)

Add object code to a pending link operation.
"""
function add_data!(link::CuLink, name::String, data::Vector{UInt8})
    cuLinkAddData(link, JIT_INPUT_OBJECT, pointer(data), length(data), name, 0, C_NULL, C_NULL)

    return nothing
end

"""
    add_file!(link::CuLink, path::String, typ::CUjitInputType)

Add data from a file to a link operation. The argument `typ` indicates the type of the
contained data.
"""
function add_file!(link::CuLink, path::String, typ::CUjitInputType)
    cuLinkAddFile(link, typ, path, 0, C_NULL, C_NULL)

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

    res = unsafe_cuLinkComplete(link, cubin_ref, size_ref)
    if res == CUDA_ERROR_NO_BINARY_FOR_GPU || res == CUDA_ERROR_INVALID_IMAGE
        options = decode(link.optionKeys, link.optionVals)
        throw(CuError(res, options[JIT_ERROR_LOG_BUFFER]))
    elseif res != SUCCESS
        throw_api_error(res)
    end

    @debug begin
        options = decode(link.optionKeys, link.optionVals)
        if isempty(options[JIT_INFO_LOG_BUFFER])
            """JIT info log is empty"""
        else
            """JIT info log:
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
