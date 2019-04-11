# EXCLUDE FROM TESTING

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

include("reduce.jl")

CUDAnative.initialize()
const dev = device()
const cap = capability(dev)
@assert(cap >= v"3.0", "this example requires a newer GPU")

len = 10^7
input = ones(Int32, len)
output = similar(input)


## CPU

benchmark_cpu = @benchmarkable begin
        reduce(+, input)
    end

@show run(benchmark_cpu)



## CUDAnative

# PTX generation
open(joinpath(@__DIR__, "reduce.jl.ptx"), "w") do f
    CUDAnative.code_ptx(f, reduce_grid, Tuple{typeof(+), CuDeviceVector{Int32,AS.Global},
                                              CuDeviceVector{Int32,AS.Global}, Int32};
                        cap=cap)
end

benchmark_gpu = @benchmarkable begin
        gpu_reduce(+, gpu_input, gpu_output)
        val = Array(gpu_output)[1]
    end setup=(
        val = nothing;
        gpu_input = CuTestArray($input);
        gpu_output = CuTestArray($output)
    ) teardown=(
        gpu_input = nothing;
        gpu_output = nothing
    )

@show run(benchmark_gpu)


## CUDA

using CUDAapi
using Libdl

cd(@__DIR__) do
    toolkit = CUDAapi.find_toolkit()
    nvcc = CUDAapi.find_cuda_binary("nvcc", toolkit)
    toolchain = CUDAapi.find_toolchain(toolkit)
    flags = `-ccbin=$(toolchain.host_compiler) -arch=sm_$(cap.major)$(cap.minor)`
    run(`$nvcc $flags -ptx -o reduce.cu.ptx reduce.cu`)
    run(`$nvcc $flags -shared --compiler-options '-fPIC' -o reduce.so reduce.cu`)
end

# Entry-point wrappers
lib = Libdl.dlopen(joinpath(@__DIR__, "reduce.so"))
setup_cuda(input)    = ccall(Libdl.dlsym(lib, "setup"), Ptr{Cvoid},
                             (Ptr{Cint}, Csize_t), input, length(input))
run_cuda(state)      = ccall(Libdl.dlsym(lib, "run"), Cint,
                             (Ptr{Cvoid},), state)
teardown_cuda(state) = ccall(Libdl.dlsym(lib, "teardown"), Cvoid,
                             (Ptr{Cvoid},), state)

# Correctness check (not part of verify.jl which is meant to run during testing)
using Test
let
    cuda_state = setup_cuda(input)
    cuda_val = run_cuda(cuda_state)
    teardown_cuda(cuda_state)
    @assert cuda_val == reduce(+, input)
end

benchmark_cuda = @benchmarkable begin
        val = run_cuda(state)
    end setup=(
        val = nothing;
        state = setup_cuda($input);
    ) teardown=(
        teardown_cuda(state)
    )

@show run(benchmark_cuda)
