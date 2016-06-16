using CUDAdrv
using Base.Test

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModuleFile(joinpath(Base.source_dir(), "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(Float32, dims)

len = prod(dims)
cudacall(vadd, len, 1, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), d_a, d_b, d_c)
c = Array(d_c)
@test_approx_eq a+b c

destroy(ctx)
