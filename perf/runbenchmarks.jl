# benchmark suite execution and codespeed submission
using Pkg
Pkg.add(url="https://github.com/christiangnrd/GPUArrays.jl", rev="akreduce")

using CUDA

using BenchmarkTools

using StableRNGs
rng = StableRNG(123)

# convenience macro to create a benchmark that requires synchronizing the GPU
macro async_benchmarkable(ex...)
    quote
        @benchmarkable CUDA.@sync blocking=true $(ex...)
    end
end

# before anything else, run latency benchmarks. these spawn subprocesses, so we don't want
# to do so after regular benchmarks have caused the memory allocator to reserve memory.
@info "Running latency benchmarks"
latency_results = include("latency.jl")

SUITE = BenchmarkGroup()

include("cuda.jl")
include("kernel.jl")
include("array.jl")

@info "Preparing main benchmarks"
warmup(SUITE; verbose=false)
tune!(SUITE)

# reclaim memory that might have been used by the tuning process
GC.gc(true)
CUDA.reclaim()

# benchmark groups that aren't part of the suite
addgroup!(SUITE, "integration")

@info "Running main benchmarks"
results = run(SUITE, verbose=true)

# integration tests (that do nasty things, so need to be run last)
@info "Running integration benchmarks"
integration_results = BenchmarkGroup()
integration_results["volumerhs"] = include("volumerhs.jl")
integration_results["byval"] = include("byval.jl")
integration_results["cudadevrt"] = include("cudadevrt.jl")

results["latency"] = latency_results
results["integration"] = integration_results

# write out the results
BenchmarkTools.save("benchmarkresults.json", median(results))
