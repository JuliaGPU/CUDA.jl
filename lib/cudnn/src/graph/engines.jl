struct UnsupportedGraphError <: Exception
    msg::String
end
Base.showerror(io::IO, e::UnsupportedGraphError) = print(io, e.msg)

# `true` for errors signalling that cuDNN cannot handle a graph, either because no engine
# finalized or because a plan reported NOT_SUPPORTED at execution time. Callers with an
# equivalent fallback may catch these; anything else indicates a bug and must propagate.
graph_unsupported(e) = e isa UnsupportedGraphError ||
                       (e isa CUDNNError && is_unsupported(e))

@public UnsupportedGraphError, graph_unsupported

function keep_engine_config(cfg::cudnnBackendDescriptor, g::Graph;
                            deterministic::Bool, math_mode)
    engine = engine_descriptor(cfg)
    try
        notes = engine_numerical_notes(engine)
        deterministic && CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC in notes && return false

        has_fp32_io = any(t -> !t.virtual && t.dtype == CUDNN_DATA_FLOAT, g.tensors)
        if math_mode == CUDACore.PEDANTIC_MATH && has_fp32_io
            CUDNN_NUMERICAL_NOTE_TENSOR_CORE in notes && return false
        end
        if math_mode != CUDACore.FAST_MATH
            CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS in notes && return false
        end
        return true
    finally
        unsafe_destroy!(engine)
    end
end

function select_plan(g::Graph, opgraph::cudnnBackendDescriptor;
                     heuristics=(CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK),
                     deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                     max_workspace::Union{Nothing,Integer}=nothing)
    deviceprop = backend_deviceprop()
    cfgs = cudnnBackendDescriptor[]
    try
        for mode in heuristics
            append!(cfgs, engine_configs(opgraph; deviceprop, mode))
        end
        for cfg in cfgs
            keep_engine_config(cfg, g; deterministic, math_mode) || continue
            plan = try_execution_plan(cfg; deviceprop)
            plan === nothing && continue
            ws = Int(plan_workspace_size(plan))
            if max_workspace !== nothing && ws > max_workspace
                unsafe_destroy!(plan)
                continue
            end
            return plan, ws
        end
        throw(UnsupportedGraphError("cuDNN: no supported engine for graph " *
                                    graph_signature(g) * " ($(length(cfgs)) candidate configs)"))
    finally
        for cfg in cfgs
            unsafe_destroy!(cfg)
        end
        unsafe_destroy!(deviceprop)
    end
end

function build!(g::Graph; kwargs...)
    g.plan === nothing || unsafe_destroy!(g)
    validate!(g)
    assign_uids!(g)
    opgraph, intermediates = lower_graph(g)
    try
        plan, workspace_size = select_plan(g, opgraph; kwargs...)
        g.plan = plan
        g.workspace_size = workspace_size
        g.variant_tensors = [t for t in g.tensors if !t.virtual]
        return g
    finally
        for d in Iterators.reverse(intermediates)
            unsafe_destroy!(d)
        end
    end
end

function is_supported(g::Graph; kwargs...)
    try
        build!(g; kwargs...)
        return true
    catch e
        graph_unsupported(e) || rethrow()
        return false
    end
end
