using CUDAnative, CUDAdrv
using Base.Test

@test devcount() > 0

include("compilation.jl")

dev = CuDevice(0)
if capability(dev) < v"2.0"
    warn("native execution not supported on SM < 2.0")
else
    include("native.jl")
end
