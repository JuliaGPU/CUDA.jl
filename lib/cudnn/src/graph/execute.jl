checked_array_pointer(t::Tensor, value) = throw(ArgumentError(
    "binding for $(t.name) must be a StridedCuArray, or a type with its own " *
    "checked_array_pointer method; got $(typeof(value))"))

# Public for array types whose memory layout the dense comparison cannot express.
# Implementations must validate whatever consistency their storage admits and
# return a device pointer.
@public checked_array_pointer

function checked_array_pointer(t::Tensor, a::StridedCuArray)
    cudnnDataType(eltype(a)) == t.dtype ||
        throw(ArgumentError("binding for $(t.name) has eltype $(eltype(a)), expected $(t.dtype)"))
    if t.ragged_offset !== nothing
        # the offsets locate each token run on the device: check the layout within a token,
        # and that the tokens are back to back, as the tensor's strides place them
        token = maximum(last, memory_layout(t.dims, t.strides); init=1)
        within(dims, strides) = filter(x -> last(x) < token, memory_layout(dims, strides))
        within(size(a), strides(a)) == within(t.dims, t.strides) &&
            view_strides(a) == collect(tuple_strides(size(a))) &&
            length(a) % token == 0 ||
            throw(DimensionMismatch(
                "binding for ragged $(t.name) has size $(size(a)), which does not hold " *
                "tokens laid out as the tensor's $(Tuple(t.dims)) with strides " *
                "$(Tuple(t.strides))"))
    elseif t.reordering == CUDNN_TENSOR_REORDERING_NONE
        memory_layout(size(a), strides(a)) == memory_layout(t.dims, t.strides) ||
            throw(DimensionMismatch(
                "binding for $(t.name) has size $(size(a)) with strides $(strides(a)), " *
                "which does not lay out the tensor's $(Tuple(t.dims)) with strides $(Tuple(t.strides))"))
    else
        length(a) == prod(t.dims) || throw(DimensionMismatch(
            "binding for $(t.name) has $(length(a)) elements, expected $(prod(t.dims))"))
    end
    return pointer(a)
end

function checked_scalar_pointer(t::Tensor, value, refs)
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
        if t.by_value
            push!(pointers, checked_scalar_pointer(t, value, refs))
        else
            push!(arrays, value)
            push!(pointers, checked_array_pointer(t, value))
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
