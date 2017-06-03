using CUDAdrv
using Base.Test

using Compat

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

len = prod(dims)
cudacall(vadd, len, 1, Tuple{DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}}, d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c

destroy!(ctx)
