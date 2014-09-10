function cualloc(T::Type, len::Integer)
        a = CUdeviceptr[0]
        nbytes = int(len) * sizeof(T)
        @cucall(:cuMemAlloc, (Ptr{CUdeviceptr}, Csize_t), a, nbytes)
        return CuPtr(a[1])
end

function cumemset(p::CUdeviceptr, value::Cuint, len::Integer)
        @cucall(:cuMemsetD32, (CUdeviceptr, Cuint, Csize_t), p, value, len)
end

function free(p::CuPtr)
        @cucall(:cuMemFree, (CUdeviceptr,), p.p)
end
