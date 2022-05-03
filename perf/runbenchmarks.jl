# benchmark suite execution and codespeed submission

using CUDA

using BenchmarkTools

using StableRNGs
rng = StableRNG(123)

# we only submit results when running on the master branch
real_run = get(ENV, "CODESPEED_BRANCH", nothing) == "master"
if real_run
    # to find untuned benchmarks
    BenchmarkTools.DEFAULT_PARAMETERS.evals = 0
end

# convenience macro to create a benchmark that requires synchronizing the GPU
macro async_benchmarkable(ex...)
    quote
        @benchmarkable CUDA.@sync $(ex...)
    end
end

# before anything else, run latency benchmarks. these spawn subprocesses, so we don't want
# to do so after regular benchmarks have caused the memory allocator to reserve memory.
@info "Running latency benchmarks"
latency_results = include("latency.jl")

SUITE = BenchmarkGroup()

# NOTE: don't use spaces in benchmark names (tobami/codespeed#256)

include("kernel.jl")
include("array.jl")

if real_run
    @info "Preparing main benchmarks"
    warmup(SUITE; verbose=false)
    tune!(SUITE)

    # reclaim memory that might have been used by the tuning process
    GC.gc(true)
    CUDA.reclaim()
end

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

println(results)


## comparison

# write out the results
BenchmarkTools.save(joinpath(@__DIR__, "results.json"), results)

# compare against previous results
# TODO: store these results so that we can compare when benchmarking PRs
reference_path = joinpath(@__DIR__, "reference.json")
if ispath(reference_path)
    reference = BenchmarkTools.load(reference_path)[1]
    comparison = judge(minimum(results), minimum(reference))

    println("Improvements:")
    println(improvements(comparison))

    println("Regressions:")
    println(regressions(comparison))
end


## submission

using JSON, HTTP

if real_run
    @info "Submitting to Codespeed..."

    basedata = Dict(
        "branch"        => ENV["CODESPEED_BRANCH"],
        "commitid"      => ENV["CODESPEED_COMMIT"],
        "project"       => ENV["CODESPEED_PROJECT"],
        "environment"   => ENV["CODESPEED_ENVIRONMENT"],
        "executable"    => ENV["CODESPEED_EXECUTABLE"]
    )

    # convert nested groups of benchmark to flat dictionaries of results
    flat_results = []
    function flatten(results, prefix="")
        for (key,value) in results
            if value isa BenchmarkGroup
                flatten(value, "$prefix$key/")
            else
                @assert value isa BenchmarkTools.Trial

                # codespeed reports maxima, but those are often very noisy.
                # get rid of measurements that unnecessarily skew the distribution.
                rmskew!(value)

                push!(flat_results,
                    Dict(basedata...,
                        "benchmark" => "$prefix$key",
                        "result_value" => median(value).time / 1e9,
                        "min" => minimum(value).time / 1e9,
                        "max" => maximum(value).time / 1e9))
            end
        end
    end
    flatten(results)

    HTTP.post("$(ENV["CODESPEED_SERVER"])/result/add/json/",
                ["Content-Type" => "application/x-www-form-urlencoded"],
                HTTP.URIs.escapeuri(Dict("json" => JSON.json(flat_results))))
end
