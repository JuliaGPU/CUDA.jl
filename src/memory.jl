# Raw memory management

export
    cualloc, cumemset, free

function cualloc{T, N<:Integer}(::Type{T}, len::N)
	if !isbits(T)
		throw(ArgumentError("Only bit-types are supported"))
	end
    ptr_ref = Ref{Ptr{Void}}()
    nbytes = len * sizeof(T)
    @apicall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), ptr_ref, nbytes)
    return DevicePtr{T}(reinterpret(Ptr{T}, ptr_ref[]), true)
end

cumemset(p::DevicePtr, value::Cuint, len::Integer) = 
    @apicall(:cuMemsetD32, (Ptr{Void}, Cuint, Csize_t), p.inner, value, len)

free(p::DevicePtr) = @apicall(:cuMemFree, (Ptr{Void},), p.inner)
