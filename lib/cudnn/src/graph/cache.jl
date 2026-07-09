function cached_graph(build::Function, key::Tuple)
    # also cache UnsupportedGraphErrors: callers with a fallback would otherwise pay the
    # full heuristics query on every call for a configuration cuDNN cannot handle.
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
