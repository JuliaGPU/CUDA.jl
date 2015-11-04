# Raw memory management

function cualloc(T::Type, len::Integer)
    ptr_ref = Ref{Ptr{Void}}()
    nbytes = Int(len) * sizeof(T)
    @cucall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), ptr_ref, nbytes)
    return DevicePtr{Void}(ptr_ref[], true)
end

cumemset(p::DevicePtr{Void}, value::Cuint, len::Integer) = 
    @cucall(:cuMemsetD32, (Ptr{Void}, Cuint, Csize_t), p.inner, value, len)

free(p::DevicePtr{Void}) = @cucall(:cuMemFree, (Ptr{Void},), p.inner)
