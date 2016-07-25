using CUDAdrv
using Base.Test

dev = CuDevice(0)
ctx = CuContext(dev)

link = CuLink()

addFile(link, joinpath(Base.source_dir(), "vadd_child.ptx"), CUDAdrv.PTX)
addFile(link, joinpath(Base.source_dir(), "vadd_parent.ptx"), CUDAdrv.PTX)

obj = complete(link)

md = CuModule(obj)
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(Float32, dims)

len = prod(dims)
cudacall(vadd, len, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c

destroy(link)
destroy(ctx)
