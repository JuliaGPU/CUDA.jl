struct LoweringContext
    tensor_descs::IdDict{Tensor,BackendDescriptor}
    intermediates::Vector{BackendDescriptor}
end

function track!(ctx::LoweringContext, d::BackendDescriptor)
    push!(ctx.intermediates, d)
    return d
end

cudnn_order(t::Tensor, v) = Int64[v[i] for i in t.backend_order]

function lower_tensor!(ctx::LoweringContext, t::Tensor)
    d = track!(ctx, backend_tensor(uid=t.uid, dims=cudnn_order(t, t.dims),
                                   strides=cudnn_order(t, t.strides), dtype=t.dtype,
                                   is_virtual=t.virtual, by_value=t.by_value,
                                   alignment=t.alignment))
    ctx.tensor_descs[t] = d
    return d
end

desc(ctx::LoweringContext, t::Tensor) = ctx.tensor_descs[t]

operation_graph_mode(::Operation) = CUDNN_OPERATIONGRAPH_MODE_AUTO

function operation_graph_mode(g::Graph)
    length(g.ops) == 1 && return operation_graph_mode(only(g.ops))
    any(op -> op isa ConvFpropOp || op isa ConvDgradOp || op isa ConvWgradOp, g.ops) &&
        return CUDNN_OPERATIONGRAPH_MODE_GENERIC_CONV_FUSION
    any(op -> op isa MatmulOp, g.ops) &&
        return CUDNN_OPERATIONGRAPH_MODE_GENERIC_MATMUL_FUSION
    all(op -> op isa PointwiseOp || op isa ReductionOp, g.ops) &&
        return CUDNN_OPERATIONGRAPH_MODE_GENERIC_POINTWISE_FUSION
    return CUDNN_OPERATIONGRAPH_MODE_AUTO
end

function lower_graph(g::Graph)
    ctx = LoweringContext(IdDict{Tensor,BackendDescriptor}(),
                          BackendDescriptor[])
    for t in g.tensors
        lower_tensor!(ctx, t)
    end
    op_descs = BackendDescriptor[]
    for op in g.ops
        push!(op_descs, track!(ctx, lower(op, ctx)))
    end
    graph = track!(ctx, operation_graph(op_descs; mode=operation_graph_mode(g)))
    return graph, ctx.intermediates
end
