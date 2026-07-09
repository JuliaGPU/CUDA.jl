abstract type Operation end

mutable struct Tensor
    name::String
    uid::Int64
    dims::Vector{Int64}
    strides::Vector{Int64}
    backend_order::Vector{Int}
    dtype::Union{Nothing,cudnnDataType_t}
    virtual::Bool
    by_value::Bool
    output::Bool
    alignment::Int
end

mutable struct Graph
    io_dtype::cudnnDataType_t
    intermediate_dtype::cudnnDataType_t
    compute_dtype::cudnnDataType_t
    tensors::Vector{Tensor}
    ops::Vector{Operation}
    plan::Union{Nothing,cudnnBackendDescriptor}
    workspace_size::Int
    variant_tensors::Vector{Tensor}
end

function Graph(; io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32)
    Graph(graph_dtype(io_dtype), graph_dtype(intermediate_dtype), graph_dtype(compute_dtype),
          Tensor[], Operation[], nothing, 0, Tensor[])
end

graph_dtype(dtype::cudnnDataType_t) = dtype
graph_dtype(::Type{T}) where {T} = cudnnDataType(T)

function dense_strides(dims)
    strides = Vector{Int64}(undef, length(dims))
    s = Int64(1)
    for i in eachindex(dims)
        strides[i] = s
        s *= Int64(dims[i])
    end
    return strides
end

function canonical_strides(dims, strides)
    out = collect(Int64, strides)
    dense = dense_strides(dims)
    for i in eachindex(out)
        dims[i] == 1 && (out[i] = dense[i])
    end
    return out
end

function tensor!(g::Graph; dims, strides=dense_strides(dims), dtype=nothing,
                 virtual::Bool=false, by_value::Bool=false, name::String="",
                 uid::Integer=0, alignment::Integer=16, output::Bool=false,
                 backend_order=nothing)
    dims_v = collect(Int64, dims)
    order = backend_order === nothing ? collect(length(dims_v):-1:1) : collect(Int, backend_order)
    sort(order) == collect(1:length(dims_v)) ||
        throw(ArgumentError("backend_order must be a permutation of tensor dimensions"))
    t = Tensor(name, Int64(uid), dims_v, canonical_strides(dims_v, strides), order,
               dtype === nothing ? nothing : graph_dtype(dtype), virtual, by_value, output,
               Int(alignment))
    push!(g.tensors, t)
    return t
end

function tensor!(g::Graph, a::DenseCuArray; name::String="", uid::Integer=0,
                 virtual::Bool=false, output::Bool=false, alignment::Integer=pointer_alignment(a),
                 backend_order=nothing)
    tensor!(g; dims=size(a), strides=strides(a), dtype=eltype(a), virtual, output,
            name, uid, alignment, backend_order)
end

function scalar!(g::Graph, ::Type{T}; rank::Integer, name::String="", uid::Integer=0) where {T}
    tensor!(g; dims=fill(Int64(1), rank), strides=fill(Int64(1), rank), dtype=T,
            by_value=true, name, uid, alignment=sizeof(T))
end

function output!(t::Tensor)
    t.virtual = false
    t.output = true
    return t
end

# look up a tensor by name, e.g. to bind tensors of a cached graph, or ones created
# implicitly by an op factory (such as the "MaskValue" fill of a causal SDPA mask)
function tensor(g::Graph, name::AbstractString)
    found = nothing
    for t in g.tensors
        t.name == name || continue
        found === nothing || error("cuDNN graph has multiple tensors named $name")
        found = t
    end
    found === nothing && error("cuDNN graph has no tensor named $name")
    return found
end

function validate!(g::Graph)
    seen = Set{Int64}()
    for t in g.tensors
        isempty(t.dims) && throw(ArgumentError("cuDNN graph tensor $(t.name) has no dims"))
        length(t.dims) == length(t.strides) ||
            throw(ArgumentError("cuDNN graph tensor $(t.name) has mismatched dims and strides"))
        all(x -> x >= 1, t.dims) ||
            throw(ArgumentError("cuDNN graph tensor $(t.name) has non-positive dims"))
        t.dtype === nothing && (t.dtype = t.virtual ? g.intermediate_dtype : g.io_dtype)
        if t.uid != 0
            t.uid in seen && throw(ArgumentError("duplicate cuDNN graph tensor uid $(t.uid)"))
            push!(seen, t.uid)
        end
    end
    return g
end

function assign_uids!(g::Graph)
    used = Set(t.uid for t in g.tensors if t.uid != 0)
    next_uid = Int64(1)
    for t in g.tensors
        t.uid != 0 && continue
        while next_uid in used
            next_uid += 1
        end
        t.uid = next_uid
        push!(used, next_uid)
    end
    return g
end

function graph_signature(g::Graph)
    parts = String[]
    for op in g.ops
        push!(parts, string(typeof(op).name.name))
    end
    return isempty(parts) ? "<empty>" : join(parts, ", ")
end

function unsafe_destroy!(g::Graph)
    g.plan === nothing || unsafe_destroy!(g.plan)
    g.plan = nothing
    empty!(g.variant_tensors)
    return
end

function pointer_alignment(a::DenseCuArray)
    p = UInt(pointer(a))
    p == 0 && return 1
    return Int(min(UInt(32), UInt(1) << min(trailing_zeros(p), 5)))
end

@public Graph, Tensor, tensor!, tensor, scalar!, output!, build!, execute!, is_supported
