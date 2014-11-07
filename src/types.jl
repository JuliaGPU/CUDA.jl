# CUDA related types

typealias CuPtr Ptr{Void}

CuPtr() = Ptr{Void}(0)

isnull(p::CuPtr) = (p == 0)
