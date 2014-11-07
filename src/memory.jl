function cualloc(T::Type, len::Integer)
        a = CuPtr[0]
        nbytes = int(len) * sizeof(T)
        @cucall(:cuMemAlloc, (Ptr{CuPtr}, Csize_t), a, nbytes)
        return CuPtr(a[1])
end

function cumemset(p::CuPtr, value::Cuint, len::Integer)
        @cucall(:cuMemsetD32, (CuPtr, Cuint, Csize_t), p, value, len)
end

function free(p::CuPtr)
        @cucall(:cuMemFree, (CuPtr,), p)
end
