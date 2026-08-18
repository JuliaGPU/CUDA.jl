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
                                   alignment=t.alignment, reordering=t.reordering))
    ctx.tensor_descs[t] = d
    return d
end

desc(ctx::LoweringContext, t::Tensor) = ctx.tensor_descs[t]

# AUTO leaves the mode attribute unset so the backend detects the pattern
# family itself; an explicit mode scopes matching to one family and can only
# remove engines from consideration, so it is opt-in via build!'s mode kwarg
function lower_graph(g::Graph;
                     mode::cudnnBackendOperationGraphMode_t=CUDNN_OPERATIONGRAPH_MODE_AUTO)
    ctx = LoweringContext(IdDict{Tensor,BackendDescriptor}(),
                          BackendDescriptor[])
    for t in g.tensors
        lower_tensor!(ctx, t)
    end
    op_descs = BackendDescriptor[]
    for op in g.ops
        push!(op_descs, track!(ctx, lower(op, ctx)))
    end
    graph = track!(ctx, operation_graph(op_descs; mode))
    return graph, ctx.intermediates
end
