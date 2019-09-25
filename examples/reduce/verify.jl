using Test

include("reduce.jl")

if capability(device()) < v"3.0"
    @warn("this example requires a newer GPU")
    exit(0)
end

len = 10^7
input = ones(Int32, len)
output = similar(input)

# CPU
cpu_val = reduce(+, input)

# CUDAnative
let
    gpu_input = CuArray(input)
    gpu_output = CuArray(output)
    gpu_reduce(+, gpu_input, gpu_output)
    gpu_val = Array(gpu_output)[1]
    @assert cpu_val == gpu_val
end
