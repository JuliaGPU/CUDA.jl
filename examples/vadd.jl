using CUDAdrv, CuArrays

using Test

md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

len = prod(dims)
cudacall(vadd, Tuple{CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}}, d_a, d_b, d_c; threads=len)

c = Array(d_c)

@test a+b ≈ c
