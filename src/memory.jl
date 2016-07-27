# Raw memory management

import Base: copy!

export
    cualloc, cumemset, free

function cualloc{T, N<:Integer}(::Type{T}, len::N=1)
    if T.abstract || !T.isleaftype
        throw(ArgumentError("Cannot allocate pointer to abstract or non-leaf type"))
    end
    ptr_ref = Ref{Ptr{Void}}()
    nbytes = len * sizeof(T)
    if nbytes <= 0
        throw(ArgumentError("Cannot allocate $nbytes bytes of memory"))
    end
    @apicall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), ptr_ref, nbytes)
    return unsafe_convert(DevicePtr{T}, reinterpret(Ptr{T}, ptr_ref[]))
end

cumemset(p::DevicePtr, value::Cuint, len::Integer) = 
    @apicall(:cuMemsetD32, (Ptr{Void}, Cuint, Csize_t), p.inner, value, len)

free(p::DevicePtr) = @apicall(:cuMemFree, (Ptr{Void},), p.inner)

function copy!{T}(dst::DevicePtr{T}, src::T)
    if !Base.datatype_pointerfree(T)
        # TODO: recursive copy?
        throw(ArgumentError("Only pointer-free types can be copied"))
    end
    @apicall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                            dst.inner, pointer_from_objref(src), sizeof(T))
    return nothing
end