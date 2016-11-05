# Raw memory management

export
    Mem

module Mem

using CUDAdrv
import CUDAdrv: @apicall


## pointer-based

"""Allocates `bytesize` bytes of linear memory on the device and returns a pointer to the
allocated memory. The allocated memory is suitably aligned for any kind of variable. The
memory is not cleared."""
function alloc(bytesize::Integer)
    bytesize == 0 && throw(ArgumentError("invalid amount of memory requested"))

    ptr_ref = Ref{Ptr{Void}}()
    @apicall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), ptr_ref, bytesize)

    return Base.unsafe_convert(DevicePtr{Void}, ptr_ref[])
end

function free(p::DevicePtr)
    @apicall(:cuMemFree, (Ptr{Void},), p.ptr)
end

set(p::DevicePtr, value::Cuint, len::Integer) =
    @apicall(:cuMemsetD32, (Ptr{Void}, Cuint, Csize_t), p.ptr, value, len)

# NOTE: upload/download also accept Ref (with Ptr <: Ref)
#       as there exists a conversion from Ref to Ptr{Void}

function upload(dst::DevicePtr, src::Ref, nbytes::Integer)
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            dst.ptr, src, nbytes)
end

function download(src::Ref, dst::DevicePtr, nbytes::Integer)
    @apicall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                            src, dst.ptr, nbytes)
end

function transfer(src::DevicePtr, dst::DevicePtr, nbytes::Integer)
    @apicall(:cuMemcpyDtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            src.ptr, dst.ptr, nbytes)
end


## object-based

function alloc{T}(::Type{T}, len::Integer=1)
    if T.abstract || !T.isleaftype
        throw(ArgumentError("cannot represent abstract or non-leaf type"))
    end
    sizeof(T) == 0 && throw(ArgumentError("cannot represent ghost types"))

    bytesize = len * sizeof(T)
    return convert(DevicePtr{T}, alloc(bytesize))
end

"""
Upload a Julia object to device memory.

Note this does only copy the object itself, and does not peek through the object in order to
get access the underlying data like `Ref` does. Consequently, this functionality should not
be used to transfer arrays, see `CuArray` for that.
"""
function upload{T}(dst::DevicePtr{T}, src::T, len::Integer=1)
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot transfer non-ptrfree objects"))
    upload(dst, Base.RefValue(src), len*sizeof(T))
end

"""
Download a Julia object from device memory.

See `upload` for notes on how arguments are processed.
"""
function download{T}(src::DevicePtr{T}, len::Integer=1)
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot transfer non-ptrfree objects"))
    dst = RefValue{T}()
    download(dst, src, len*sizeof(T))
    return dst[]
end

end
