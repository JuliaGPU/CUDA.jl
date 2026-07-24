function cached_graph(build::Function, key::Tuple)
    # Cache unsupported graphs to avoid repeated heuristic searches.
    cached = get!(handle().plans, key) do
        try
            build()
        catch e
            e isa UnsupportedGraphError || rethrow()
            e
        end
    end
    cached isa UnsupportedGraphError && throw(cached)
    return cached::Graph
end

@public cached_graph
