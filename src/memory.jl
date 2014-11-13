# Raw memory management

function cualloc(T::Type, len::Integer)
    dptr_box = ptrbox(DevicePtr{Void})
    nbytes = int(len) * sizeof(T)
    @cucall(:cuMemAlloc, (Ptr{DevicePtr{Void}}, Csize_t), dptr_box, nbytes)
    return ptrunbox(dptr_box)
end

cumemset(p::DevicePtr{Void}, value::Cuint, len::Integer) = 
    @cucall(:cuMemsetD32, (DevicePtr{Void}, Cuint, Csize_t), p, value, len)

free(p::DevicePtr{Void}) = @cucall(:cuMemFree, (DevicePtr{Void},), p)
