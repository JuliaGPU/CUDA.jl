function checked_array_pointer(t::Tensor, a::DenseCuArray)
    size(a) == Tuple(t.dims) ||
        throw(DimensionMismatch("binding for $(t.name) has size $(size(a)), expected $(Tuple(t.dims))"))
    canonical_strides(t.dims, strides(a)) == t.strides ||
        throw(DimensionMismatch("binding for $(t.name) has strides $(strides(a)), expected $(Tuple(t.strides))"))
    cudnnDataType(eltype(a)) == t.dtype ||
        throw(ArgumentError("binding for $(t.name) has eltype $(eltype(a)), expected $(juliaDataType(t.dtype))"))
    return pointer(a)
end

function checked_scalar_pointer(t::Tensor, value, refs)
    t.by_value || throw(ArgumentError("binding for $(t.name) must be a DenseCuArray"))
    T = juliaDataType(t.dtype)
    ref = value isa Ref ? value : Ref{T}(convert(T, value))
    push!(refs, ref)
    return Base.unsafe_convert(Ptr{T}, ref)
end

function execute!(g::Graph, bindings::AbstractDict)
    g.plan === nothing && throw(ArgumentError("cuDNN graph must be built before execute!"))

    pointers = Any[]
    arrays = Any[]
    refs = Any[]
    for t in g.variant_tensors
        haskey(bindings, t) || throw(ArgumentError("missing binding for cuDNN graph tensor $(t.name)"))
        value = bindings[t]
        if value isa DenseCuArray
            push!(arrays, value)
            push!(pointers, checked_array_pointer(t, value))
        else
            push!(pointers, checked_scalar_pointer(t, value, refs))
        end
    end

    uids = Int64[t.uid for t in g.variant_tensors]
    with_workspace(g.workspace_size) do workspace
        ws = sizeof(workspace) == 0 ? C_NULL : pointer(workspace)
        vp = variant_pack(uids=uids, pointers=pointers, workspace=ws)
        try
            GC.@preserve g arrays refs vp begin
                cudnnBackendExecute(handle(), g.plan.ptr, vp.ptr)
            end
        finally
            unsafe_destroy!(vp)
        end
    end
    return g
end

execute!(g::Graph, bindings::Pair...) = execute!(g, IdDict{Tensor,Any}(bindings...))
