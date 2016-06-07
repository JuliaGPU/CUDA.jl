using CUDAnative
using Base.Test

dev = CuDevice(0)
ctx = CuContext(dev)

len = 60

a = rand(Float32, len)
b = rand(Float32, len)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(Float32, len)

# @inline/@noinline doesn't seem to matter
add(a,b) = a+b

@target ptx function map_inner{F}(fun::F, a, b, c)
    i = blockIdx().x + (threadIdx().x-1) * gridDim().x
    c[i] = fun(a[i], b[i])

    return nothing
end

# BUG: the generated code only has 3 arguments (with `fun` probably being
#      inlined, or similar). This means we'll be passing invalid parameters to
#      the driver.
code_llvm(map_inner, (typeof(add), CuDeviceArray{Float32},
                      CuDeviceArray{Float32}, CuDeviceArray{Float32}))

@cuda (len, 1) map_inner(add, d_a, d_b, d_c)

c = Array(d_c)
@test a+b == c
