function cualloc(T::Type, len::Integer)
        a = DevicePtr{Void}[0]
        nbytes = int(len) * sizeof(T)
        @cucall(:cuMemAlloc, (Ptr{DevicePtr{Void}}, Csize_t), a, nbytes)
        return DevicePtr{Void}(a[1])
end

function cumemset(p::DevicePtr{Void}, value::Cuint, len::Integer)
        @cucall(:cuMemsetD32, (DevicePtr{Void}, Cuint, Csize_t), p, value, len)
end

function free(p::DevicePtr{Void})
        @cucall(:cuMemFree, (DevicePtr{Void},), p)
end
