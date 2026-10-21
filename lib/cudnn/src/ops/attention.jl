# where the sequences are: a padded batch, optionally masked by per-element lengths,
# or packed back to back along the token dimension
struct PaddedSequences{Q<:Union{Nothing,DenseCuArray{Int32,4}},
                       KV<:Union{Nothing,DenseCuArray{Int32,4}}}
    seq_len_q::Q
    seq_len_kv::KV
end

struct PackedSequences{Q<:DenseCuArray{Int32,1},KV<:DenseCuArray{Int32,1}}
    cu_seq_len_q::Q
    cu_seq_len_kv::KV
    max_seq_len_q::Int
    max_seq_len_kv::Int
end

# whether the sequences are delimited by padding masks
masked(seqs::PaddedSequences) = seqs.seq_len_q !== nothing
masked(::PackedSequences) = true

function check_attention_args(fn, T, dropout_p, bias)
    T in (Float16, BFloat16) ||
        throw(ArgumentError("cuDNN $fn only supports Float16/BFloat16, got $T"))
    dropout_p == 0 || throw(ArgumentError("$fn does not support dropout"))
    bias === nothing || throw(ArgumentError("$fn does not support bias"))
    return
end

function check_head_dim(d)
    d % 8 == 0 || throw(ArgumentError("head dimension must be a multiple of 8, got $d"))
    d <= 256 || throw(ArgumentError("head dimension must be <= 256, got $d"))
    return
end

attention_args_supported(T, d) = T in (Float16, BFloat16) && d % 8 == 0 && d <= 256

function check_attention_heads(q, k, v, out)
    d, h = size(q)
    dk, hk = size(k)
    dv, hv = size(v)
    dk == d || throw(DimensionMismatch("k must have head dimension $d, got $dk"))
    dv == d || throw(DimensionMismatch("v must have head dimension $d, got $dv"))
    hk == hv || throw(DimensionMismatch("k and v head counts must match"))
    h % hk == 0 || throw(DimensionMismatch("q heads must be a multiple of k/v heads"))
    size(out) == size(q) ||
        throw(DimensionMismatch("out must have size $(size(q)), got $(size(out))"))
    return d, h, hk
end

# (d, h, hk, sq, skv, b), with sq and skv the bounds of packed sequences
function attention_dims(seqs::PaddedSequences, q, k, v, out)
    d, h, hk = check_attention_heads(q, k, v, out)
    _, _, sq, b = size(q)
    _, _, skv, bk = size(k)
    size(v, 3) == skv || throw(DimensionMismatch("k and v sequence lengths must match"))
    bk == b && size(v, 4) == b ||
        throw(DimensionMismatch("q, k, and v batch sizes must match"))
    (seqs.seq_len_q === nothing) == (seqs.seq_len_kv === nothing) ||
        throw(ArgumentError("seq_len_q and seq_len_kv must be passed together"))
    if masked(seqs)
        size(seqs.seq_len_q) == (1, 1, 1, b) ||
            throw(DimensionMismatch("seq_len_q must have size $((1, 1, 1, b)), " *
                                    "got $(size(seqs.seq_len_q))"))
        size(seqs.seq_len_kv) == (1, 1, 1, b) ||
            throw(DimensionMismatch("seq_len_kv must have size $((1, 1, 1, b)), " *
                                    "got $(size(seqs.seq_len_kv))"))
    end
    return d, h, hk, sq, skv, b
end

function attention_dims(seqs::PackedSequences, q, k, v, out)
    d, h, hk = check_attention_heads(q, k, v, out)
    size(v, 3) == size(k, 3) || throw(DimensionMismatch("k and v token counts must match"))
    n = length(seqs.cu_seq_len_q)
    n >= 2 || throw(ArgumentError("cu_seq_len_q must hold at least two offsets"))
    length(seqs.cu_seq_len_kv) == n || throw(DimensionMismatch(
        "cu_seq_len_q and cu_seq_len_kv must have the same length, got $n and " *
        "$(length(seqs.cu_seq_len_kv))"))
    sq, skv = seqs.max_seq_len_q, seqs.max_seq_len_kv
    sq >= 1 && skv >= 1 ||
        throw(ArgumentError("max_seq_len_q and max_seq_len_kv must be positive"))
    return d, h, hk, sq, skv, n - 1
end

# stats uses the (1, s, h, b) packing the backward's spill buffers require;
# the forward writes it through strides at no cost
function attention_stats_dims(stats, h, sq, b)
    stats === nothing && return
    size(stats) == (1, sq, h, b) ||
        throw(DimensionMismatch("stats must have size $((1, sq, h, b)), got $(size(stats))"))
    return
end

function attention_gradient_dims(dq, dk, dv, dO, q, k, v)
    size(dO) == size(q) ||
        throw(DimensionMismatch("dO must have size $(size(q)), got $(size(dO))"))
    size(dq) == size(q) ||
        throw(DimensionMismatch("dQ must have size $(size(q)), got $(size(dq))"))
    size(dk) == size(k) ||
        throw(DimensionMismatch("dK must have size $(size(k)), got $(size(dk))"))
    size(dv) == size(v) ||
        throw(DimensionMismatch("dV must have size $(size(v)), got $(size(dv))"))
    return
end

attention_optional_key(a) =
    a === nothing ? nothing : (eltype(a), size(a), strides(a), pointer_alignment(a))

# By-value scale does not affect plan selection.
attention_key(seqs::PaddedSequences, dims, arrays) =
    (:padded, map(a -> (size(a), strides(a), pointer_alignment(a)), arrays)...,
     attention_optional_key(seqs.seq_len_q), attention_optional_key(seqs.seq_len_kv))
# Total token counts stay out of the key: they do not shape the plan.
attention_key(::PackedSequences, dims, arrays) =
    (:packed, dims..., map(pointer_alignment, arrays)...)

# under bottom-right alignment without padding masks, bind_sequences! binds the by-value
# lengths sdpa_fwd! and sdpa_bwd! create instead
function sequence_tensors!(g, seqs, b)
    masked(seqs) || return nothing, nothing
    return tensor!(g; dims=(1, 1, 1, b), dtype=Int32, name="SeqLenQ"),
           tensor!(g; dims=(1, 1, 1, b), dtype=Int32, name="SeqLenKV")
end

# the token offsets, per sequence, of packed (d, nh, tokens) buffers
ragged_offsets!(g, ::PaddedSequences, b) = nothing, nothing
ragged_offsets!(g, ::PackedSequences, b) =
    tensor!(g; dims=(1, 1, 1, b + 1), dtype=Int64, name="RaggedOffsetQ"),
    tensor!(g; dims=(1, 1, 1, b + 1), dtype=Int64, name="RaggedOffsetKV")

# the padding masks and element offsets of the sequences delimited by cu
packed_lengths(cu) = reshape(view(cu, 2:length(cu)) .- view(cu, 1:length(cu)-1), 1, 1, 1, :)
packed_offsets(cu, token) = cu .* Int64(token)

function bind_sequences!(bindings, g, seqs::PaddedSequences, (d, h, hk, sq, skv, b),
                         causal)
    if masked(seqs)
        bindings[tensor(g, "SeqLenQ")] = seqs.seq_len_q
        bindings[tensor(g, "SeqLenKV")] = seqs.seq_len_kv
    elseif causal === :bottom_right
        bindings[tensor(g, "MaskSeqLenQ")] = Int32(sq)
        bindings[tensor(g, "MaskSeqLenKV")] = Int32(skv)
    end
    return
end

function bind_sequences!(bindings, g, seqs::PackedSequences, (d, h, hk, sq, skv, b),
                         causal)
    bindings[tensor(g, "RaggedOffsetQ")] = packed_offsets(seqs.cu_seq_len_q, d * h)
    bindings[tensor(g, "RaggedOffsetKV")] = packed_offsets(seqs.cu_seq_len_kv, d * hk)
    bindings[tensor(g, "SeqLenQ")] = packed_lengths(seqs.cu_seq_len_q)
    bindings[tensor(g, "SeqLenKV")] = packed_lengths(seqs.cu_seq_len_kv)
    return
end

function build_attention_graph(seqs, (d, h, hk, sq, skv, b), out, q, k, v, stats, causal;
                               deterministic, math_mode, max_workspace)
    g = Graph(io_dtype=eltype(q), intermediate_dtype=Float32, compute_dtype=Float32)
    offq, offkv = ragged_offsets!(g, seqs, b)
    io(name, a, nh, s, offset; output=false) = offset === nothing ?
        tensor!(g, a; name, output) :
        tensor!(g; dims=(d, nh, s, b), strides=(1, d, d * nh, d * nh), dtype=eltype(a),
                name, output, alignment=pointer_alignment(a), ragged_offset=offset)
    tq = io("Q", q, h, sq, offq)
    tk = io("K", k, hk, skv, offkv)
    tv = io("V", v, hk, skv, offkv)
    to = io("O", out, h, sq, offq; output=true)
    ts = scalar!(g, Float32; rank=4, name="Scale")
    # the (1, sq, h, b) stats buffer, declared on sdpa_fwd!'s (1, h, sq, b) dims
    tstats = stats === nothing ? nothing :
             tensor!(g; dims=(1, h, sq, b), strides=(1, sq, 1, sq * h),
                     dtype=Float32, name="Stats", output=true)
    tseqq, tseqkv = sequence_tensors!(g, seqs, b)
    sdpa_fwd!(g, tq, tk, tv; o=to, scale=ts, stats=tstats, seq_len_q=tseqq,
              seq_len_kv=tseqkv, causal)
    build!(g; deterministic, math_mode, max_workspace)
end

function build_attention_backward_graph(seqs, (d, h, hk, sq, skv, b), dq, dk, dv, dO,
                                        q, k, v, o, causal;
                                        deterministic, math_mode, max_workspace)
    # as cudnn-frontend does; there is no engine for it on SM12x
    seqs isa PackedSequences && deterministic && capability(device()).major in (8, 12) &&
        throw(UnsupportedGraphError(
            "cuDNN: deterministic packed attention backward is not supported on " *
            "compute capability $(capability(device()))"))
    T = eltype(q)
    g = Graph(io_dtype=T, intermediate_dtype=Float32, compute_dtype=Float32)
    offq, offkv = ragged_offsets!(g, seqs, b)
    # canonical (d, h, s, b) buffers bind through transposed strides for free;
    # packed ones are located per sequence by the ragged offsets
    io(name, a, s, nh, offset; output=false) = offset === nothing ?
        tensor!(g; dims=(d, s, nh, b), strides=(1, d * nh, d, d * nh * s), dtype=T,
                output, name) :
        tensor!(g; dims=(d, s, nh, b), strides=(1, d * nh, d, d * nh), dtype=T, name,
                output, alignment=pointer_alignment(a), ragged_offset=offset)
    tq = io("Q", q, sq, h, offq)
    tk = io("K", k, skv, hk, offkv)
    tv = io("V", v, skv, hk, offkv)
    to = io("O", o, sq, h, offq)
    tdO = io("dO", dO, sq, h, offq)
    tstats = tensor!(g; dims=(1, sq, h, b), dtype=Float32, name="Stats")
    tdq = io("dQ", dq, sq, h, offq; output=true)
    tdk = io("dK", dk, skv, hk, offkv; output=true)
    tdv = io("dV", dv, skv, hk, offkv; output=true)
    ts = scalar!(g, Float32; rank=4, name="Scale")
    tseqq, tseqkv = sequence_tensors!(g, seqs, b)
    sdpa_bwd!(g, tq, tk, tv, to, tdO, tstats; dQ=tdq, dK=tdk, dV=tdv, scale=ts,
              seq_len_q=tseqq, seq_len_kv=tseqkv, causal, deterministic)
    build!(g; deterministic, math_mode, max_workspace)
end

function attention_plan(seqs, out, q, k, v, stats, causal;
                        deterministic, math_mode, max_workspace)
    dims = attention_dims(seqs, q, k, v, out)
    _, h, _, sq, _, b = dims
    attention_stats_dims(stats, h, sq, b)
    key = (:attention_fwd, eltype(q), attention_key(seqs, dims, (q, k, v, out))...,
           attention_optional_key(stats), causal, deterministic, math_mode, max_workspace)
    g = cached_graph(key) do
        build_attention_graph(seqs, dims, out, q, k, v, stats, causal;
                              deterministic, math_mode, max_workspace)
    end
    return g, dims
end

function attention_backward_plan(seqs, dq, dk, dv, dO, q, k, v, o, stats, causal;
                                 deterministic, math_mode, max_workspace)
    dims = attention_dims(seqs, q, k, v, o)
    _, h, _, sq, _, b = dims
    attention_gradient_dims(dq, dk, dv, dO, q, k, v)
    attention_stats_dims(stats, h, sq, b)
    key = (:attention_bwd, eltype(q),
           attention_key(seqs, dims, (dq, dk, dv, dO, q, k, v, o, stats))..., causal,
           deterministic, math_mode, max_workspace)
    g = cached_graph(key) do
        build_attention_backward_graph(seqs, dims, dq, dk, dv, dO, q, k, v, o, causal;
                                       deterministic, math_mode, max_workspace)
    end
    return g, dims
end

# the scratch outputs the composite pattern requires: dense in sdpa_bwd!'s (d, s, h, b)
# order when padded, in the token layout of the gradients when packed. A deterministic
# graph has no dQ accumulator.
function bind_backward_workspaces!(bindings, g, seqs, (d, h, hk, sq, skv, b), q, k,
                                   deterministic)
    T = eltype(q)
    bindings[tensor(g, "SoftmaxSum")] = CUDACore.zeros(Float32, 1, sq, h, b)
    if seqs isa PaddedSequences
        deterministic ||
            (bindings[tensor(g, "dQAccum")] = CUDACore.zeros(Float32, d, sq, h, b))
        hk == h && return
        bindings[tensor(g, "dKFullhead")] = CUDACore.zeros(T, d, skv, h, b)
        bindings[tensor(g, "dVFullhead")] = CUDACore.zeros(T, d, skv, h, b)
    else
        deterministic ||
            (bindings[tensor(g, "dQAccum")] = CUDACore.zeros(Float32, d, h, size(q, 3)))
        hk == h && return
        bindings[tensor(g, "dKFullhead")] = CUDACore.zeros(T, d, h, size(k, 3))
        bindings[tensor(g, "dVFullhead")] = CUDACore.zeros(T, d, h, size(k, 3))
        bindings[tensor(g, "FullheadRaggedOffset")] =
            packed_offsets(seqs.cu_seq_len_kv, d * h)
    end
    return
end

function run_attention!(seqs, out::DenseCuArray{T}, q::DenseCuArray{T},
                        k::DenseCuArray{T}, v::DenseCuArray{T};
                        scale::Real=1/sqrt(size(q, 1)),
                        causal::Union{Bool,Symbol}=false, dropout_p::Real=0, bias=nothing,
                        stats::Union{Nothing,DenseCuArray{Float32,4}}=nothing,
                        deterministic::Bool=false,
                        math_mode=CUDACore.math_mode(),
                        max_workspace::Union{Nothing,Integer}=nothing) where {T}
    check_attention_args("attention!", T, dropout_p, bias)
    causal = sdpa_causal(causal)
    isempty(out) && return out
    check_head_dim(size(q, 1))
    g, dims = attention_plan(seqs, out, q, k, v, stats, causal;
                             deterministic, math_mode, max_workspace)

    bindings = IdDict{Tensor,Any}(
        tensor(g, "Q") => q,
        tensor(g, "K") => k,
        tensor(g, "V") => v,
        tensor(g, "O") => out,
        tensor(g, "Scale") => Float32(scale),
    )
    stats === nothing || (bindings[tensor(g, "Stats")] = stats)
    causal === :none || (bindings[tensor(g, "MaskValue")] = Float32(-Inf))
    bind_sequences!(bindings, g, seqs, dims, causal)
    execute!(g, bindings)
    return out
end

function run_attention_backward!(seqs, dq::DenseCuArray{T}, dk::DenseCuArray{T},
                                 dv::DenseCuArray{T}, dO::DenseCuArray{T},
                                 q::DenseCuArray{T}, k::DenseCuArray{T},
                                 v::DenseCuArray{T}, o::DenseCuArray{T},
                                 stats::DenseCuArray{Float32,4};
                                 scale::Real=1/sqrt(size(q, 1)),
                                 causal::Union{Bool,Symbol}=false, dropout_p::Real=0,
                                 bias=nothing,
                                 deterministic::Bool=false,
                                 math_mode=CUDACore.math_mode(),
                                 max_workspace::Union{Nothing,Integer}=nothing) where {T}
    check_attention_args("attention_backward!", T, dropout_p, bias)
    causal = sdpa_causal(causal)
    isempty(dq) && isempty(dk) && isempty(dv) && return dq, dk, dv
    check_head_dim(size(q, 1))
    g, dims = attention_backward_plan(seqs, dq, dk, dv, dO, q, k, v, o, stats, causal;
                                      deterministic, math_mode, max_workspace)

    bindings = IdDict{Tensor,Any}(
        tensor(g, "Q") => q,
        tensor(g, "K") => k,
        tensor(g, "V") => v,
        tensor(g, "O") => o,
        tensor(g, "dO") => dO,
        tensor(g, "Stats") => stats,
        tensor(g, "dQ") => dq,
        tensor(g, "dK") => dk,
        tensor(g, "dV") => dv,
        tensor(g, "Scale") => Float32(scale),
        tensor(g, "One") => 1f0,
    )
    bind_backward_workspaces!(bindings, g, seqs, dims, q, k, deterministic)
    (causal !== :none || masked(seqs)) &&
        (bindings[tensor(g, "MaskValue")] = Float32(-Inf))
    bind_sequences!(bindings, g, seqs, dims, causal)
    execute!(g, bindings)
    return dq, dk, dv
end

function plan_supported(plan)
    try
        plan()
        return true
    catch e
        graph_unsupported(e) || rethrow()
        return false
    end
end

function attention_plan_supported(seqs, out::DenseCuArray{T}, q::DenseCuArray{T},
                                  k::DenseCuArray{T}, v::DenseCuArray{T};
                                  causal::Union{Bool,Symbol}=false,
                                  stats::Union{Nothing,DenseCuArray{Float32,4}}=nothing,
                                  deterministic::Bool=false,
                                  math_mode=CUDACore.math_mode(),
                                  max_workspace::Union{Nothing,Integer}=nothing) where {T}
    causal = sdpa_causal(causal)
    attention_args_supported(T, size(q, 1)) || return false
    isempty(out) && return true
    plan_supported() do
        attention_plan(seqs, out, q, k, v, stats, causal;
                       deterministic, math_mode, max_workspace)
    end
end

function attention_backward_plan_supported(seqs, dq::DenseCuArray{T}, dk::DenseCuArray{T},
                                           dv::DenseCuArray{T}, dO::DenseCuArray{T},
                                           q::DenseCuArray{T}, k::DenseCuArray{T},
                                           v::DenseCuArray{T}, o::DenseCuArray{T},
                                           stats::DenseCuArray{Float32,4};
                                           causal::Union{Bool,Symbol}=false,
                                           deterministic::Bool=false,
                                           math_mode=CUDACore.math_mode(),
                                           max_workspace::Union{Nothing,Integer}=nothing) where {T}
    causal = sdpa_causal(causal)
    attention_args_supported(T, size(q, 1)) || return false
    isempty(dq) && isempty(dk) && isempty(dv) && return true
    plan_supported() do
        attention_backward_plan(seqs, dq, dk, dv, dO, q, k, v, o, stats, causal;
                                deterministic, math_mode, max_workspace)
    end
end

## padded (d, h, s, b) or packed (d, h, tokens) arrays

padded_sequences(; seq_len_q::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                 seq_len_kv::Union{Nothing,DenseCuArray{Int32,4}}=nothing, kwargs...) =
    PaddedSequences(seq_len_q, seq_len_kv), kwargs
packed_sequences(; cu_seq_len_q::DenseCuArray{Int32,1},
                 cu_seq_len_kv::DenseCuArray{Int32,1},
                 max_seq_len_q::Integer, max_seq_len_kv::Integer, kwargs...) =
    PackedSequences(cu_seq_len_q, cu_seq_len_kv, Int(max_seq_len_q),
                    Int(max_seq_len_kv)), kwargs

function attention!(out::DenseCuArray{T,4}, q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                    v::DenseCuArray{T,4}; kwargs...) where {T}
    seqs, kwargs = padded_sequences(; kwargs...)
    run_attention!(seqs, out, q, k, v; kwargs...)
end

function attention!(out::DenseCuArray{T,3}, q::DenseCuArray{T,3}, k::DenseCuArray{T,3},
                    v::DenseCuArray{T,3}; kwargs...) where {T}
    seqs, kwargs = packed_sequences(; kwargs...)
    run_attention!(seqs, out, q, k, v; kwargs...)
end

function attention_backward!(dq::DenseCuArray{T,4}, dk::DenseCuArray{T,4},
                             dv::DenseCuArray{T,4}, dO::DenseCuArray{T,4},
                             q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                             v::DenseCuArray{T,4}, o::DenseCuArray{T,4},
                             stats::DenseCuArray{Float32,4}; kwargs...) where {T}
    seqs, kwargs = padded_sequences(; kwargs...)
    run_attention_backward!(seqs, dq, dk, dv, dO, q, k, v, o, stats; kwargs...)
end

function attention_backward!(dq::DenseCuArray{T,3}, dk::DenseCuArray{T,3},
                             dv::DenseCuArray{T,3}, dO::DenseCuArray{T,3},
                             q::DenseCuArray{T,3}, k::DenseCuArray{T,3},
                             v::DenseCuArray{T,3}, o::DenseCuArray{T,3},
                             stats::DenseCuArray{Float32,4}; kwargs...) where {T}
    seqs, kwargs = packed_sequences(; kwargs...)
    run_attention_backward!(seqs, dq, dk, dv, dO, q, k, v, o, stats; kwargs...)
end

function attention_supported(out::DenseCuArray{T,4}, q::DenseCuArray{T,4},
                             k::DenseCuArray{T,4}, v::DenseCuArray{T,4};
                             kwargs...) where {T}
    seqs, kwargs = padded_sequences(; kwargs...)
    attention_plan_supported(seqs, out, q, k, v; kwargs...)
end

function attention_supported(out::DenseCuArray{T,3}, q::DenseCuArray{T,3},
                             k::DenseCuArray{T,3}, v::DenseCuArray{T,3};
                             kwargs...) where {T}
    seqs, kwargs = packed_sequences(; kwargs...)
    attention_plan_supported(seqs, out, q, k, v; kwargs...)
end

function attention_backward_supported(dq::DenseCuArray{T,4}, dk::DenseCuArray{T,4},
                                      dv::DenseCuArray{T,4}, dO::DenseCuArray{T,4},
                                      q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                                      v::DenseCuArray{T,4}, o::DenseCuArray{T,4},
                                      stats::DenseCuArray{Float32,4}; kwargs...) where {T}
    seqs, kwargs = padded_sequences(; kwargs...)
    attention_backward_plan_supported(seqs, dq, dk, dv, dO, q, k, v, o, stats; kwargs...)
end

function attention_backward_supported(dq::DenseCuArray{T,3}, dk::DenseCuArray{T,3},
                                      dv::DenseCuArray{T,3}, dO::DenseCuArray{T,3},
                                      q::DenseCuArray{T,3}, k::DenseCuArray{T,3},
                                      v::DenseCuArray{T,3}, o::DenseCuArray{T,3},
                                      stats::DenseCuArray{Float32,4}; kwargs...) where {T}
    seqs, kwargs = packed_sequences(; kwargs...)
    attention_backward_plan_supported(seqs, dq, dk, dv, dO, q, k, v, o, stats; kwargs...)
end

function attention_backward(dO::DenseCuArray{T,N}, q::DenseCuArray{T,N},
                            k::DenseCuArray{T,N}, v::DenseCuArray{T,N},
                            o::DenseCuArray{T,N}, stats::DenseCuArray{Float32,4};
                            kwargs...) where {T,N}
    dq, dk, dv = similar(q), similar(k), similar(v)
    attention_backward!(dq, dk, dv, dO, q, k, v, o, stats; kwargs...)
end

function attention(q::DenseCuArray{T,N}, k::DenseCuArray{T,N}, v::DenseCuArray{T,N};
                   kwargs...) where {T,N}
    out = similar(q)
    attention!(out, q, k, v; kwargs...)
end

@doc raw"""
    attention(q, k, v; kwargs...)
    attention!(out, q, k, v; kwargs...)
    attention_backward(dO, q, k, v, o, stats; kwargs...)
    attention_backward!(dQ, dK, dV, dO, q, k, v, o, stats; kwargs...)
    attention_supported(out, q, k, v; kwargs...) -> Bool
    attention_backward_supported(dQ, dK, dV, dO, q, k, v, o, stats; kwargs...) -> Bool

Execute fused scaled dot-product attention with tensors shaped
`(head_dim, heads, sequence_length, batch_size)`, except `stats`, which is
shaped `(1, sequence_length, heads, batch_size)`.

Supported inputs are `Float16` and `BFloat16` `DenseCuArray`s. `scale` defaults to
`1 / sqrt(head_dim)`. Forward can write the Float32 `stats` tensor needed by backward.
Dense padding masks are enabled by passing Int32 `seq_len_q` and `seq_len_kv` tensors
shaped `(1, 1, 1, batch_size)`; padded backward gradients are defined inside the valid
region only. `dropout_p` must be zero and `bias` must be `nothing`.

`causal=false` (or `:none`) disables causal masking. `causal=true` (or `:top_left`)
aligns the causal diagonal with the top-left corner of the score matrix: query `i`
attends to keys `j ≤ i`. `causal=:bottom_right` aligns it with the bottom-right corner,
for queries that are the last positions of the key sequence (KV-cache decoding, chunked
prefill): query `i` attends to keys `j ≤ i + seq_len_kv - seq_len_q`, with the lengths
taken per batch element when they are passed and from the array sizes otherwise. Without
`seq_len_q` and `seq_len_kv`, it is unsupported when there are more queries than keys;
with them, the output of queries that see no key is undefined.

Packed variable-length sequences use `(head_dim, heads, tokens)` arrays instead, holding
the sequences back to back, with the required keywords `cu_seq_len_q`, `cu_seq_len_kv`,
`max_seq_len_q` and `max_seq_len_kv`. The first two are Int32 vectors of `batch_size + 1`
nondecreasing token offsets starting at 0: sequence `b` holds tokens
`cu_seq_len_q[b]+1:cu_seq_len_q[b+1]`. The other two bound the sequence lengths, and
`stats` is padded to `(1, max_seq_len_q, heads, batch_size)`. The offsets stay on the
device and are not checked: a last offset beyond the token count, or a sequence longer
than its bound, accesses memory out of bounds. Plans depend on the bounds and the batch
size but not on the token counts, so rounding the bounds up lets batches share them.
Causal masks align with each sequence's own lengths, and tokens past the last offset are
ignored. `deterministic=true` is unsupported for the packed backward on compute
capability 8.x and 12.x.

The backward pass builds the composite graph of `sdpa_bwd!` and allocates its
workspace buffers per call: a Float32 dQ accumulator and a softmax sum, and two more
under grouped-query attention. `deterministic=true` replaces the dQ accumulator, which
sums in arbitrary order, with a plan workspace of about `2 * batch_size * heads *
seq_len_q * seq_len_kv` bytes; `max_workspace` bounds it.

The support predicates build and cache a plan without executing it.
"""
attention, attention!, attention_backward, attention_backward!, attention_supported,
attention_backward_supported

@public attention, attention!, attention_backward, attention_backward!,
        attention_supported, attention_backward_supported
