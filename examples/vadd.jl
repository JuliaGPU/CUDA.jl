using CUDAnative
using Base.Test

@target ptx function kernel_vadd(a, b, c)
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    c[i] = a[i] + b[i]

    return nothing
end

dev = CuDevice(0)
ctx = CuContext(dev)

dims = (3,4)
a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)
c = Array(Float32, dims)

len = prod(dims)
@cuda (len,1) kernel_vadd(CuIn(a), CuIn(b), CuOut(c))
@test_approx_eq a+b c

destroy(ctx)
