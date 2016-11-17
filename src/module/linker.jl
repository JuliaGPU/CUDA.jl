# Linking of different PTX modules

export
    CuLink, complete,
    addData, addFile


typealias CuLinkState_t Ptr{Void}

type CuLink
    handle::CuLinkState_t
    ctx::CuContext

    options::Dict{CUjit_option,Any}
    optionKeys::Vector{CUjit_option}
    optionVals::Vector{Ptr{Void}}

    function CuLink()
        handle_ref = Ref{CuLinkState_t}()

        options = Dict{CUjit_option,Any}()
        options[ERROR_LOG_BUFFER] = Array(UInt8, 1024*1024)
        @static if DEBUG
            options[GENERATE_LINE_INFO] = true
            options[GENERATE_DEBUG_INFO] = true

            options[INFO_LOG_BUFFER] = Array(UInt8, 1024*1024)
            options[LOG_VERBOSE] = true
        end
        optionKeys, optionVals = encode(options)

        @apicall(:cuLinkCreate, (Cuint, Ptr{CUjit_option}, Ptr{Ptr{Void}}, Ptr{CuLinkState_t}),
                                length(optionKeys), optionKeys, optionVals, handle_ref)

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx, options, optionKeys, optionVals)
        block_finalizer(obj, ctx)
        finalizer(obj, finalize)
        return obj
    end
end

function finalize(link::CuLink)
    trace("Finalizing CuLink at $(Base.pointer_from_objref(link))")
    @apicall(:cuLinkDestroy, (CuLinkState_t,), link)
    unblock_finalizer(link, link.ctx)
end

Base.unsafe_convert(::Type{CuLinkState_t}, link::CuLink) = link.handle

Base.:(==)(a::CuLink, b::CuLink) = a.handle == b.handle
Base.hash(link::CuLink, h::UInt) = hash(link.handle, h)


# wrapper type for tracking ownership (as `data` is invalidated when the linker is destroyed)
immutable CuLinkImage
    data::Array{UInt8}
    link::CuLink
end

CuModule(img::CuLinkImage, args...) = CuModule(img.data, args...)


"""
Complete a pending linker invocation, returning an output image.
"""
function complete(link::CuLink)
    cubin_ref = Ref{Ptr{Void}}()
    size_ref = Ref{Csize_t}()

    try
        @apicall(:cuLinkComplete, (CuLinkState_t, Ptr{Ptr{Void}}, Ptr{Csize_t}),
                                  link, cubin_ref, size_ref)
    catch err
        (err == ERROR_NO_BINARY_FOR_GPU || err == ERROR_INVALID_IMAGE) || rethrow(err)
        options = decode(link.optionKeys, link.optionVals)
        rethrow(CuError(err.code, options[ERROR_LOG_BUFFER]))
    end

    @static if DEBUG
        options = decode(link.optionKeys, link.optionVals)
        if isempty(options[INFO_LOG_BUFFER])
            debug("JIT info log is empty")
        else
            debug("JIT info log: ", repr_indented(options[INFO_LOG_BUFFER]))
        end
    end

    data = unsafe_wrap(Array, convert(Ptr{UInt8}, cubin_ref[]), size_ref[])
    return CuLinkImage(data, link)
end

function addData(link::CuLink, name::String, data::Union{Vector{UInt8},String}, typ::CUjit_input)
    # NOTE: ccall can't directly convert String to Ptr{Void}, so step through a typed Ptr
    if typ == PTX
        # additionally, in the case of PTX there shouldn't be any embedded NULLs
        raw_data = Base.unsafe_convert(Cstring, Base.cconvert(Cstring, String(data)))
    else
        raw_data = Base.unsafe_convert(Vector{UInt8}, Base.cconvert(Vector{UInt8}, data))
    end
    typed_ptr = pointer(raw_data)
    untyped_ptr = convert(Ptr{Void}, typed_ptr)

    @apicall(:cuLinkAddData,
             (CuLinkState_t, CUjit_input, Ptr{Void}, Csize_t, Cstring, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Void}}),
             link, typ, untyped_ptr, length(data), name, 0, C_NULL, C_NULL)

    return nothing
end

function addFile(link::CuLink, path::String, typ::CUjit_input)
    @apicall(:cuLinkAddFile,
             (CuLinkState_t, CUjit_input, Cstring, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Void}}),
             link, typ, path, 0, C_NULL, C_NULL)

    return nothing
end
