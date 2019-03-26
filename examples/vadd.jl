using CUDAdrv

using Test

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = Mem.alloc(Mem.Device, sizeof(a))
d_b = Mem.alloc(Mem.Device, sizeof(a))
d_c = Mem.alloc(Mem.Device, sizeof(c))

copyto!(d_a, pointer(a), sizeof(a))
copyto!(d_b, pointer(b), sizeof(b))

len = prod(dims)
cudacall(vadd, Tuple{CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}}, d_a, d_b, d_c; threads=len)

copyto!(pointer(c), d_c, sizeof(c))

@test a+b ≈ c

destroy!(ctx)
