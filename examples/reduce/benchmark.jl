include("reduce.jl")

dev = CuDevice(0)
@assert(capability(dev) >= v"3.0", "this implementation requires a newer GPU")

lib = Libdl.dlopen(joinpath(@__DIR__, "reduce.so"))
setup_cuda(input)    = ccall(Libdl.dlsym(lib, "setup"), Ptr{Void},
                             (Ptr{Cint}, Csize_t), input, length(input))
run_cuda(state)      = ccall(Libdl.dlsym(lib, "run"), Cint,
                             (Ptr{Void},), state)
teardown_cuda(state) = ccall(Libdl.dlsym(lib, "teardown"), Void,
                             (Ptr{Void},), state)


#
# Correctness
#

using Base.Test

len = 10^6
input = ones(Int32,len)

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

# CUDA

let
    cuda_state = setup_cuda(input)
    cuda_val = run_cuda(cuda_state)
    teardown_cuda(cuda_state)
    @assert cpu_val == cuda_val
end


#
# Performance
#

using BenchmarkTools


## CUDAnative

ctx = CuContext(dev)
benchmark_gpu = @benchmarkable begin
        gpu_reduce(+, gpu_input, gpu_output)
        val = Array(gpu_output)[1]
    end setup=(
        val = nothing;
        gpu_input = CuArray($input);
        gpu_output = similar(gpu_input)
    ) teardown=(
        @assert val == $cpu_val;
        gpu_input = nothing;
        gpu_output = nothing;
        gc()
    )
println(run(benchmark_gpu))
destroy(ctx)


## CUDA

benchmark_cuda = @benchmarkable begin
        val = run_cuda(state)
    end setup=(
        val = nothing;
        state = setup_cuda($input);
    ) teardown=(
        teardown_cuda(state)
    )
println(run(benchmark_cuda))
