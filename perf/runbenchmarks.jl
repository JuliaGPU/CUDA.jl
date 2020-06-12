# benchmark suite execution and codespeed submission

using CUDA, BenchmarkTools

SUITE = BenchmarkGroup()

include("array.jl")

warmup(SUITE)
results = run(SUITE)


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
