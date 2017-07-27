# Raw memory management

export
    Mem

module Mem

using CUDAdrv
import CUDAdrv: @apicall, OwnedPtr


## pointer-based

# TODO: single copy function, with `memcpykind(Ptr, Ptr)` (cfr. CUDArt)?

"""
    alloc(bytes::Integer)

Allocates `bytesize` bytes of linear memory on the device and returns a pointer to the
allocated memory. The allocated memory is suitably aligned for any kind of variable. The
memory is not cleared, use [`free(::OwnedPtr)`](@ref) for that.
"""
function alloc(bytesize::Integer)
    bytesize == 0 && throw(ArgumentError("invalid amount of memory requested"))

    ptr_ref = Ref{Ptr{Void}}()
    @apicall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), ptr_ref, bytesize)

    return OwnedPtr{Void}(ptr_ref[], CuCurrentContext())
end

"""
    free(p::OwnedPtr)

Frees device memory.
"""
function free(p::OwnedPtr)
    @apicall(:cuMemFree, (Ptr{Void},), p.ptr)
end


"""
    info()

Returns a tuple of two integers, indicating respectively the free and total amount of memory
(in bytes) available for allocation by the CUDA context.
"""
function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    @apicall(:cuMemGetInfo, (Ptr{Csize_t},Ptr{Csize_t}), free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

"""
    free()

Returns the free amount of memory (in bytes), available for allocation by the CUDA context.
"""
free() = info()[1]

"""
    total()

Returns the total amount of memory (in bytes), available for allocation by the CUDA context.
"""
total() = info()[2]

"""
    used()

Returns the used amount of memory (in bytes), allocated by the CUDA context.
"""
used() = total()-free()



"""
    set(p::OwnedPtr, value::Cuint, len::Integer)

Initializes device memory, copying the value `val` for `len` times.
"""
set(p::OwnedPtr, value::Cuint, len::Integer) =
    @apicall(:cuMemsetD32, (Ptr{Void}, Cuint, Csize_t), p.ptr, value, len)

# NOTE: upload/download also accept Ref (with Ptr <: Ref)
#       as there exists a conversion from Ref to Ptr{Void}

"""
    upload(dst::OwnedPtr, src, nbytes::Integer)

Upload `nbytes` memory from `src` at the host to `dst` on the device.
"""
function upload(dst::OwnedPtr, src::Ref, nbytes::Integer)
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            dst.ptr, src, nbytes)
end

"""
    download(dst::OwnedPtr, src, nbytes::Integer)

Download `nbytes` memory from `src` on the device to `src` on the host.
"""
function download(dst::Ref, src::OwnedPtr, nbytes::Integer)
    @apicall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                            dst, src.ptr, nbytes)
end

"""
    download(dst::OwnedPtr, src, nbytes::Integer)

Transfer `nbytes` of device memory from `src` to `dst`.
"""
function transfer(dst::OwnedPtr, src::OwnedPtr, nbytes::Integer)
    @apicall(:cuMemcpyDtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            dst.ptr, src.ptr, nbytes)
end


## object-based

# TODO: varargs functions for uploading multiple objects at once?
"""
    alloc{T}(len=1)

Allocates space for `len` objects of type `T` on the device and returns a pointer to the
allocated memory. The memory is not cleared, use [`free(::OwnedPtr)`](@ref) for that.
"""
function alloc{T}(::Type{T}, len::Integer=1)
    @static if VERSION >= v"0.6.0-dev.2123"
        if isa(T, UnionAll) || T.abstract || !T.isleaftype
            throw(ArgumentError("cannot represent abstract or non-leaf type"))
        end
    else
        # 0.5 compatibility
        if T.abstract || !T.isleaftype
            throw(ArgumentError("cannot represent abstract or non-leaf type"))
        end
    end
    sizeof(T) == 0 && throw(ArgumentError("cannot represent ghost types"))

    return convert(OwnedPtr{T}, alloc(len*sizeof(T)))
end

"""
    upload{T}(src::T)
    upload{T}(dst::OwnedPtr{T}, src::T)

Upload an object `src` from the host to the device. If a destination `dst` is not provided,
new memory is allocated and uploaded to.

Note this does only upload the object itself, and does not peek through it in order to get
to the underlying data (like `Ref` does). Consequently, this functionality should not be
used to transfer eg. arrays, use [`CuArray`](@ref)'s [`copy!`](@ref) functionality for that.
"""
function upload{T}(dst::OwnedPtr{T}, src::T)
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot transfer non-ptrfree objects"))
    upload(dst, Base.RefValue(src), sizeof(T))
end

function upload{T}(src::T)
    dst = alloc(T)
    upload(dst, Base.RefValue(src), sizeof(T))
    return dst
end

"""
    download{T}(src::OwnedPtr{T})

Download an object `src` from the device and return it as a host object.

See [`upload`](@ref) for notes on how arguments are processed.
"""
function download{T}(src::OwnedPtr{T})
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot transfer non-ptrfree objects"))
    dst = Base.RefValue{T}()
    download(dst, src, sizeof(T))
    return dst[]
end

end
