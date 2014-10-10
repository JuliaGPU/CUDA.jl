module __ptx__GPUModule

using CUDA

export vadd

function vadd(a, b, c)
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    c[i] = a[i] + b[i]

    return nothing
end

end

using CUDA, Base.Test
using __ptx__GPUModule

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

@cuda (__ptx__GPUModule, len, 1) vadd(CuIn(a), CuIn(b), CuOut(c))
c = to_host(gc)

free(ga)
free(gb)
free(gc)

@test c == (a + b)
