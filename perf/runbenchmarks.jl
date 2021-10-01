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
        # use non-blocking sync to reduce overhead
        @benchmarkable CUDA.@sync blocking=false $(ex...)
    end
end

SUITE = BenchmarkGroup()

# NOTE: don't use spaces in benchmark names (tobami/codespeed#256)

include("kernel.jl")
include("array.jl")

if real_run
    warmup(SUITE; verbose=false)
    tune!(SUITE)

    # reclaim memory that might have been used by the tuning process
    GC.gc(true)
    CUDA.reclaim()
end

# latency benchmarks spawn external processes and take very long,
# so don't benefit from warm-up or tuning.
include("latency.jl")

# integration tests are currently not part of the benchmark suite
addgroup!(SUITE, "integration")

@info "Running benchmarks"
results = run(SUITE, verbose=true)

# integration tests (that do nasty things, so need to be run last)
results["integration"]["volumerhs"] = include("volumerhs.jl")
results["integration"]["byval"] = include("byval.jl")
results["integration"]["cudadevrt"] = include("cudadevrt.jl")

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
