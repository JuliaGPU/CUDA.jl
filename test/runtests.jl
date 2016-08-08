using CUDAnative, CUDAdrv
using Base.Test

@test devcount() > 0

include("compilation.jl")
include("codegen.jl")

dev = CuDevice(0)
if capability(dev) < v"2.0"
    warn("native execution not supported on SM < 2.0")
else
    ctx = CuContext(dev)

    include("execution.jl")
    include("array.jl")
    include("intrinsics.jl")

    destroy(ctx)
end
