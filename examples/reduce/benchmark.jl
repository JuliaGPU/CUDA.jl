# EXCLUDE FROM TESTING
# this example requires BenchmarkTools, which can be hard to install

using BenchmarkTools
include("reduce.jl")

ctx = CuCurrentContext()
dev = device(ctx)
@assert(capability(dev) >= v"3.0", "this example requires a newer GPU")

len = 10^7
input = ones(Int32, len)


## CUDAnative

# PTX generation
open(joinpath(@__DIR__, "reduce.jl.ptx"), "w") do f
    code_ptx(f, reduce_grid, Tuple{typeof(+), CuDeviceVector{Int32,AS.Global},
                                   CuDeviceVector{Int32,AS.Global}, Int32};
                        cap=v"6.1.0")
end

benchmark_gpu = @benchmarkable begin
        gpu_reduce(+, gpu_input, gpu_output)
        val = Array(gpu_output)[1]
    end setup=(
        val = nothing;
        gpu_input = CuArray($input);
        gpu_output = similar(gpu_input)
    ) teardown=(
        gpu_input = nothing;
        gpu_output = nothing;
        gc()
    )
println(run(benchmark_gpu))


## CUDA

# Entry-point wrappers
lib = Libdl.dlopen(joinpath(@__DIR__, "reduce.so"))
setup_cuda(input)    = ccall(Libdl.dlsym(lib, "setup"), Ptr{Cvoid},
                             (Ptr{Cint}, Csize_t), input, length(input))
run_cuda(state)      = ccall(Libdl.dlsym(lib, "run"), Cint,
                             (Ptr{Cvoid},), state)
teardown_cuda(state) = ccall(Libdl.dlsym(lib, "teardown"), Void,
                             (Ptr{Cvoid},), state)

# Correctness check (not part of verify.jl which is meant to run during testing)
using Compat
using Compat.Test
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
println(run(benchmark_cuda))
