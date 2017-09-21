using CUDAdrv

using Compat
using Compat.Test

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

x = similar(a)

d_a = Mem.upload(a)
d_b = Mem.upload(b)
d_c = Mem.alloc(c)

len = prod(dims)
cudacall(vadd, len, 1, Tuple{Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}}, d_a, d_b, d_c)

Mem.download(c, d_c)
@test a+b ≈ c

destroy!(ctx)
