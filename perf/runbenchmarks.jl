# benchmark suite execution and codespeed submission

using CUDA
CUDA.allowscalar(false)

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.evals = 0     # to find untuned benchmarks

using StableRNGs
rng = StableRNG(123)

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

@info "Warming up"
warmup(SUITE; verbose=false)

paramsfile = joinpath(first(DEPOT_PATH), "datadeps", "CUDA_benchmark_params.json")
# NOTE: using a path that survives across CI runs
mkpath(dirname(paramsfile))
if !isfile(paramsfile)
    @warn "No saved parameters found, tuning all benchmarks"
    tune!(SUITE)
else
    loadparams!(SUITE, BenchmarkTools.load(paramsfile)[1], :evals, :samples)

    # find untuned benchmarks for which we have the default evals==0
    function find_untuned(group::BenchmarkGroup, untuned=Dict(), prefix="")
        for (name, b) in group
            find_untuned(b, untuned, isempty(prefix) ? name : "$prefix/$name")
        end
        return untuned
    end
    function find_untuned(b::BenchmarkTools.Benchmark, untuned=Dict(), prefix="")
        if params(b).evals == 0
            untuned[prefix] = b
        end
        return untuned
    end
    untuned = find_untuned(SUITE)

    if !isempty(untuned)
        @info "Re-tuning the following benchmarks: $(join(keys(untuned), ", "))"
        foreach(tune!, values(untuned))
    end
end
BenchmarkTools.save(paramsfile, params(SUITE))

# reclaim memory that might have been used by the tuning process
GC.gc(true)
CUDA.reclaim()

# latency benchmarks spawn external processes and take very long,
# so don't benefit from warm-up or tuning.
include("latency.jl")

@info "Running benchmarks"
results = run(SUITE, verbose=true)
println(results)


## submission

using JSON, HTTP

if get(ENV, "CODESPEED_BRANCH", nothing) == "master"
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
