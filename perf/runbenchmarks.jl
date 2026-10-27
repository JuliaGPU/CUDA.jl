# benchmark suite execution and codespeed submission

using CUDA

# Record the environment alongside benchmark results.
CUDA.versioninfo()

using BenchmarkTools, Statistics, JSON

using Random, StableRNGs
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

@info "Running main benchmarks"
results = run(SUITE, verbose=true)

# integration tests (that do nasty things, so need to be run last)
@info "Running integration benchmarks"
integration_results = BenchmarkGroup()
integration_results["volumerhs"] = include("volumerhs.jl")
integration_results["byval"] = include("byval.jl")
integration_results["cudadevrt"] = include("cudadevrt.jl")

# the first part of every name is its category in the benchmark history: everything
# compiled (performance), fresh processes (latency), or larger codes (applications).
# see perf/benchmark-config.json
results = BenchmarkGroup("performance" => results, "latency" => latency_results,
                         "applications" => integration_results)

# every statistic of every benchmark, and the environment they were measured in, for the
# benchmark history at https://juliagpu.org/perf/. times are in nanoseconds.
# the history compares the minimum rather than the median (perf/benchmark-config.json): at
# the sub-microsecond scale of many of these benchmarks, OS scheduler jitter dominates the
# median and produces 5-15% trial-to-trial variance, while the minimum reflects the
# un-preempted code path and is stable to <1% across trials.
function export_results(file, results)
    benchmarks = Dict{String,Any}()
    for (keys, trial) in BenchmarkTools.leaves(results)
        times = trial.times
        benchmarks[join(keys, "/")] = Dict(
            "min" => minimum(times), "p25" => quantile(times, 0.25),
            "median" => median(times), "p75" => quantile(times, 0.75),
            "mean" => mean(times), "max" => maximum(times),
            "std" => length(times) > 1 ? std(times) : nothing,
            "gctime" => median(trial.gctimes),
            "memory" => trial.memory, "allocs" => trial.allocs,
            "samples" => length(times), "evals" => trial.params.evals)
    end
    # the versions of every package loaded, to explain changes the commit itself does not
    packages = Dict(id.name => string(pkgversion(mod)) for (id, mod) in Base.loaded_modules
                    if pkgversion(mod) !== nothing)
    env = Dict(
        "julia" => string(VERSION),
        "gpu" => CUDA.name(device()),
        # NVML reports the kernel-mode driver's own version, but is absent on Tegra
        "driver" => CUDA.has_nvml() ? string(CUDA.NVML.driver_version()) :
                                      string(CUDA.driver_version()),
        "runtime" => string(CUDA.runtime_version()),
        "toolkit" => string(CUDA.compiler_version()),
        "cpu" => Sys.cpu_info()[1].model,
        "os" => Sys.iswindows() ? "Windows" : readchomp(`uname -sr`),
        "threads" => Threads.nthreads(),
        "agent" => get(ENV, "BUILDKITE_AGENT_NAME", nothing),
        "package" => string(pkgversion(CUDA)),
        "packages" => packages)
    open(file, "w") do io
        JSON.print(io, Dict("version" => 2, "env" => env, "benchmarks" => benchmarks))
    end
end
export_results(length(ARGS) >= 1 ? ARGS[1] : "benchmarks.json", results)
