using Base.Test
include("reduce.jl")

dev = CuDevice(0)
if capability(dev) < v"3.0"
    warn("this example requires a newer GPU")
    exit(0)
end

len = 10^7
input = ones(Int32, len)

# CPU
cpu_val = reduce(+, input)

# CUDAnative
let
    ctx = CuContext(dev)
    gpu_input = CuArray(input)
    gpu_output = similar(gpu_input)
    gpu_reduce(+, gpu_input, gpu_output)
    gpu_val = Array(gpu_output)[1]
    destroy(ctx)
    @assert cpu_val == gpu_val
end
