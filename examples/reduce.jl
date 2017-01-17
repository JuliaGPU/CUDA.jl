using CUDAdrv, CUDAnative
using Base.Test

# Fast parallel reduction for Kepler hardware
#
# Based on devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
#
# FURTHER IMPROVEMENTS:
# - analyze LLVM IR for redundant Int32/Int64 conversions
#   without adding Int32() everywhere (JuliaGPU/CUDAnative.jl#25)
# - use atomic memory operations
# - add dispatch-based fallbacks for non-Kepler hardware
# - dynamic block/grid size based on device capabilities
# - improve documentation
# - vectorized memory access
#   devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

# Reduce a value across a warp
function reduce_warp{F<:Function,T}(op::F, val::T)::T
    offset = CUDAnative.warpsize() ÷ Int32(2)
    # TODO: this can be unrolled if warpsize is known...
    while offset > 0
        val = op(val, shfl_down(val, offset))
        offset ÷= Int32(2)
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
function reduce_block{F<:Function,T}(op::F, val::T)::T
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    # TODO: use fldmod1 (JuliaGPU/CUDAnative.jl#28)
    wid =  div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end

# Reduce an array across a complete grid
function reduce_grid{F<:Function,T}(op::F, input::CuDeviceArray{T,1}, output::CuDeviceArray{T,1}, N::Integer)
    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-Int32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= N
        @inbounds val = op(val, input[i])
        i += step
    end

    val = reduce_block(op, val)

    if threadIdx().x == Int32(1)
        @inbounds output[blockIdx().x] = val
    end

    return
end

"""
Reduce a large array.

Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce{F<:Function,T}(op::F, input::CuArray{T,1}, output::CuArray{T,1})
    N = length(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((N + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        throw(ArgumentError("output array too small, should be at least $blocks elements"))
    end

    @cuda (blocks,threads) reduce_grid(op, input, output, Int32(N))
    @cuda (1,1024) reduce_grid(op, output, output, Int32(blocks))

    return
end

function gpu_reduce{F<:Function,T}(op::F, input::Array{T,1})
    ctx = CuCurrentContext()
    dev = device(ctx)
    @assert(capability(dev) >= v"3.0", "this implementation requires a newer GPU")

    gpu_input = CuArray(input)
    gpu_output = similar(gpu_input)

    gpu_reduce(op, gpu_input, gpu_output)

    return Array(gpu_output)[1]
end



#
# Main
#

len = 10^6
input = ones(Int32,len)

cpu_val = reduce(+, input)

dev = CuDevice(0)
ctx = CuContext(dev)
gpu_val = gpu_reduce(+, input)
destroy(ctx)

@assert cpu_val == gpu_val



#
# Benchmark
#

using BenchmarkTools

open(joinpath(@__DIR__, "reduce.ptx"), "w") do f
    CUDAnative.code_ptx(f, reduce_grid, Tuple{typeof(+), CuDeviceArray{Int32, 1}, CuDeviceArray{Int32, 1}, Int32}; cap=v"6.1.0")
end


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

lib = Libdl.dlopen(joinpath(@__DIR__, "reduce", "reduce.so"))
setup_cuda(input) = ccall(Libdl.dlsym(lib, "setup"), Ptr{Void},
                          (Ptr{Cint}, Csize_t), input, length(input))
run_cuda(state) = ccall(Libdl.dlsym(lib, "run"), Cint,
                        (Ptr{Void},), state)
teardown_cuda(state) = ccall(Libdl.dlsym(lib, "teardown"), Void,
                             (Ptr{Void},), state)

benchmark_cuda = @benchmarkable begin
        val = run_cuda(state)
    end setup=(
        val = nothing;
        state = setup_cuda($input);
    ) teardown=(
        @assert val == $cpu_val;
        teardown_cuda(state)
    )
println(run(benchmark_cuda))
