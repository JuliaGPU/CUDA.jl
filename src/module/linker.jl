# Linking of different PTX modules

export
    CuLink, addData, addFile, complete


const CuLinkState_t = Ptr{Void}

"""
    CuLink()

Creates a pending JIT linker invocation.
"""
type CuLink
    handle::CuLinkState_t
    ctx::CuContext

    options::Dict{CUjit_option,Any}
    optionKeys::Vector{CUjit_option}
    optionVals::Vector{Ptr{Void}}

    function CuLink()
        handle_ref = Ref{CuLinkState_t}()

        options = Dict{CUjit_option,Any}()
        options[ERROR_LOG_BUFFER] = Array{UInt8}(1024*1024)
        @static if CUDAapi.DEBUG
            options[GENERATE_LINE_INFO] = true
            options[GENERATE_DEBUG_INFO] = true

            options[INFO_LOG_BUFFER] = Array{UInt8}(1024*1024)
            options[LOG_VERBOSE] = true
        end
        optionKeys, optionVals = encode(options)

        @apicall(:cuLinkCreate, (Cuint, Ptr{CUjit_option}, Ptr{Ptr{Void}}, Ptr{CuLinkState_t}),
                                length(optionKeys), optionKeys, optionVals, handle_ref)

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx, options, optionKeys, optionVals)
        finalizer(obj, unsafe_destroy!)
        return obj
    end
end

function unsafe_destroy!(link::CuLink)
    if isvalid(link.ctx)
        @trace("Finalizing CuLink object at $(Base.pointer_from_objref(link))")
        @apicall(:cuLinkDestroy, (CuLinkState_t,), link)
    else
        @trace("Skipping finalizer for CuLink object at $(Base.pointer_from_objref(link))) because context is no longer valid")
    end
end

Base.unsafe_convert(::Type{CuLinkState_t}, link::CuLink) = link.handle

Base.:(==)(a::CuLink, b::CuLink) = a.handle == b.handle
Base.hash(link::CuLink, h::UInt) = hash(link.handle, h)

# TODO: push! -- or add_data!, add_file!
"""
    addData(link::CuLink, name::String, data, type::CUjit_input)

Add data to a link operation. The argument `typ` indicates the type of the data.
"""
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

"""
    addFile(link::CuLink, path::String, typ::CUjit_input)

Add data from a file to a link operation. The argument `typ` indicates the type of the
contained data.
"""
function addFile(link::CuLink, path::String, typ::CUjit_input)
    @apicall(:cuLinkAddFile,
             (CuLinkState_t, CUjit_input, Cstring, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Void}}),
             link, typ, path, 0, C_NULL, C_NULL)

    return nothing
end

"""
The result of a linking operation.

This object keeps its parent linker object alive, as destroying a linker destroys linked
images too.
"""
immutable CuLinkImage
    data::Array{UInt8}
    link::CuLink
end


"""
    complete(link::CuLink)

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

    @static if CUDAapi.DEBUG
        options = decode(link.optionKeys, link.optionVals)
        if isempty(options[INFO_LOG_BUFFER])
            @debug("JIT info log is empty")
        else
            @debug("JIT info log: ", CUDAapi.repr_indented(options[INFO_LOG_BUFFER]))
        end
    end

    data = unsafe_wrap(Array, convert(Ptr{UInt8}, cubin_ref[]), size_ref[])
    return CuLinkImage(data, link)
end

"""
    CuModule(img::CuLinkImage, ...)
    CuModule(f::Function, img::CuLinkImage, ...)

Create a CUDA module from a completed linking operation. Options from `CuModule` apply.
"""
CuModule(img::CuLinkImage, args...) = CuModule(img.data, args...)
