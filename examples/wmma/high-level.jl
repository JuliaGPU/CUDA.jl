# Need https://github.com/JuliaLang/julia/pull/33970
# and  https://github.com/JuliaLang/julia/pull/34043
if VERSION < v"1.4.0-DEV.666"
    exit()
end

using CUDAnative, CUDAdrv
if capability(device()) < v"7.0"
    exit()
end

### START
using Test

using CUDAnative
const CuArray = CUDAnative.CuHostArray  # real applications: use CuArrays.jl

a     = rand(Float16, (16, 16))
b     = rand(Float16, (16, 16))
c     = rand(Float32, (16, 16))

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(c)
d_dev = similar(c_dev)

function kernel(a_dev, b_dev, c_dev, d_dev)
    conf = WMMA.Config{16, 16, 16, Float32}

    a_frag = WMMA.load_a(pointer(a_dev), 16, WMMA.ColMajor, conf)
    b_frag = WMMA.load_b(pointer(b_dev), 16, WMMA.ColMajor, conf)
    c_frag = WMMA.load_c(pointer(c_dev), 16, WMMA.ColMajor, conf)

    c_frag = 0.5f0 .* c_frag

    d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)

    WMMA.store_d(pointer(d_dev), d_frag, 16, WMMA.ColMajor, conf)

    return
end

@cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
d = Array(d_dev)

@test all(isapprox.(a * b + 0.5 * c, d; rtol=0.01))
### END
