using CUDA, Base.Test

@test devcount() > 0

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModule("vadd.ptx")
f = CuFunction(md, "vadd")

dims = (3, 4)
len = prod(dims)
a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(Float32, dims)

launch(f, len, 1, (a_dev, b_dev, c_dev))
c = to_host(c_dev)

free(a_dev)
free(b_dev)
free(c_dev)

@test c == (a + b)
