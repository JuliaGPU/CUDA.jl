using Test

using CUDAdrv
include(joinpath(@__DIR__, "..", "test", "array.jl"))   # real applications: use CuArrays.jl

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CuTestArray(a)
d_b = CuTestArray(b)
d_c = CuTestArray(c)

len = prod(dims)
cudacall(vadd, Tuple{CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}}, d_a, d_b, d_c; threads=len)

@test a+b ≈ Array(d_c)

CUDAdrv.unsafe_destroy!(ctx)
