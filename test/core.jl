using CUDA, Base.Test

@test devcount() > 0

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModule("vadd.ptx")
f = CuFunction(md, "vadd")

siz = (3, 4)
len = prod(siz)
a = round(rand(Float32, siz) * 100)
b = round(rand(Float32, siz) * 100)

ga = CuArray(a)
gb = CuArray(b)
gc = CuArray(Float32, siz)

launch(f, len, 1, (ga, gb, gc))
c = to_host(gc)

free(ga)
free(gb)
free(gc)

@test c == (a + b)
