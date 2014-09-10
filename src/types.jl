# CUDA related types

typealias CUdeviceptr Ptr{Void}

immutable CuPtr
        p::CUdeviceptr

        CuPtr() = new(convert(CUdeviceptr, 0))
        CuPtr(p::CUdeviceptr) = new(p)
end

cubox(p::CuPtr) = cubox(p.p)

isnull(p::CuPtr) = (p.p == 0)
