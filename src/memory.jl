# Raw memory management

export Mem


## buffer type

# forward definition in global.jl
# NOTE: this results in Buffer not being part of the Mem submodule


Base.pointer(buf::Buffer) = buf.ptr

Base.unsafe_convert{T}(::Type{Ptr{T}}, buf::Buffer) = convert(Ptr{T}, pointer(buf))

Base.isnull(buf::Buffer) = (pointer(buf) == C_NULL)



## refcounting

const refcounts = Dict{Buffer, Int}()

function refcount(buf::Buffer)
    get(refcounts, Base.unsafe_convert(Ptr{Void}, buf), 0)
end

"""
    retain(buf)

Increase the refcount of a pointer.
"""
function retain(buf::Buffer)
    refcount = get!(refcounts, buf, 0)
    refcounts[buf] = refcount + 1
    return
end

"""
    release(buf)

Decrease the refcount of a pointer. Returns `true` if the refcount has dropped to 0, and
some action needs to be taken.
"""
function release(buf::Buffer)
    haskey(refcounts, buf) || error("Release of unmanaged $buf")
    refcount = refcounts[buf]
    @assert refcount > 0 "Release of dead $buf"
    refcounts[buf] = refcount - 1
    return refcount==1
end


## memory info

module Mem

using CUDAdrv
import CUDAdrv: @apicall, Buffer

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


## pointer-based

"""
    alloc(bytes::Integer)

Allocates `bytesize` bytes of linear memory on the device and returns a pointer to the
allocated memory. The allocated memory is suitably aligned for any kind of variable. The
memory is not cleared, use [`free(::Buffer)`](@ref) for that.
"""
function alloc(bytesize::Integer)
    bytesize == 0 && throw(ArgumentError("invalid amount of memory requested"))

    ptr_ref = Ref{Ptr{Void}}()
    @apicall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), ptr_ref, bytesize)
    return Buffer(ptr_ref[], bytesize, CuCurrentContext())
end

"""
    free(buf::Buffer)

Frees device memory.
"""
function free(buf::Buffer)
    @apicall(:cuMemFree, (Ptr{Void},), pointer(buf))
    return
end

"""
    set(buf::Buffer, value::Cuint, len::Integer)

Initializes device memory, copying the value `val` for `len` times.
"""
set(buf::Buffer, value::Cuint, len::Integer) =
    @apicall(:cuMemsetD32, (Ptr{Void}, Cuint, Csize_t), pointer(buf), value, len)

# NOTE: upload/download also accept Ref (with Ptr <: Ref)
#       as there exists a conversion from Ref to Ptr{Void}

"""
    upload(dst::Buffer, src, nbytes::Integer)

Upload `nbytes` memory from `src` at the host to `dst` on the device.
"""
function upload(dst::Buffer, src::Ref, nbytes::Integer)
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            pointer(dst), src, nbytes)
end

"""
    download(dst::Buffer, src, nbytes::Integer)

Download `nbytes` memory from `src` on the device to `src` on the host.
"""
function download(dst::Ref, src::Buffer, nbytes::Integer)
    @apicall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                            dst, pointer(src), nbytes)
end

"""
    transfer(dst::Buffer, src::Buffer, nbytes::Integer)

Transfer `nbytes` of device memory from `src` to `dst`.
"""
function transfer(dst::Buffer, src::Buffer, nbytes::Integer)
    @apicall(:cuMemcpyDtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            pointer(dst), pointer(src), nbytes)
end


## object-based

# TODO: varargs functions for uploading multiple objects at once?

"""
    alloc{T}(len=1)

Allocates space for `len` objects of type `T` on the device and returns a pointer to the
allocated memory. The memory is not cleared, use [`free(::Buffer)`](@ref) for that.
"""
function alloc(::Type{T}, len::Integer=1) where T
    if isa(T, UnionAll) || T.abstract || !T.isleaftype
        throw(ArgumentError("cannot represent abstract or non-leaf type"))
    end
    sizeof(T) == 0 && throw(ArgumentError("cannot represent ghost types"))

    return alloc(len*sizeof(T))
end

"""
    upload{T}(src::T)
    upload{T}(dst::Buffer, src::T)

Upload an object `src` from the host to the device. If a destination `dst` is not provided,
new memory is allocated and uploaded to.

Note this does only upload the object itself, and does not peek through it in order to get
to the underlying data (like `Ref` does). Consequently, this functionality should not be
used to transfer eg. arrays, use [`CuArray`](@ref)'s [`copy!`](@ref) functionality for that.
"""
function upload(dst::Buffer, src::T) where T
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot transfer non-ptrfree objects"))
    upload(dst, Base.RefValue(src), sizeof(T))
end

function upload(src::T) where T
    dst = alloc(T)
    upload(dst, Base.RefValue(src), sizeof(T))
    return dst
end

"""
    download{T}(::Type{T}, src::Buffer)

Download an object `src` from the device and return it as a host object of type `T`.

See [`upload`](@ref) for notes on how arguments are processed.
"""
function download(::Type{T}, src::Buffer) where T
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot transfer non-ptrfree objects"))
    dst = Base.RefValue{T}()
    download(dst, src, sizeof(T))
    return dst[]
end

end
