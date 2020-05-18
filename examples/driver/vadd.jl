using Test

using CUDA

md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

len = prod(dims)
cudacall(vadd, Tuple{CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}}, d_a, d_b, d_c; threads=len)

@test a+b ≈ Array(d_c)
