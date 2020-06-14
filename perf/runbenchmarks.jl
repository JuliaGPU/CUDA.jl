# benchmark suite execution and codespeed submission

using CUDA, BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.evals = 0     # to find untuned benchmarks

SUITE = BenchmarkGroup()

# NOTE: don't use spaces in benchmark names (tobami/codespeed#256)

include("kernel.jl")
include("array.jl")

@info "Warming up"
warmup(SUITE; verbose=false)

paramsfile = joinpath(first(DEPOT_PATH), "cache", "CUDA_benchmark_params.json")
mkpath(dirname(paramsfile))
if !isfile(paramsfile)
    @warn "No saved parameters found, will re-tune all benchmarks"
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
        @info "Tuning parameters: $(join(keys(untuned), ", "))"
        foreach(tune!, values(untuned))
        BenchmarkTools.save(paramsfile, params(SUITE))
    end
end

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
