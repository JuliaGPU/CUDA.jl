function attention_dims(q, k, v, out)
    d, hq, sq, b = size(q)
    dk, hk, skv, bk = size(k)
    dv, hv, skvv, bv = size(v)
    dk == d || throw(DimensionMismatch("k must have head dimension $d, got $dk"))
    dv == d || throw(DimensionMismatch("v must have head dimension $d, got $dv"))
    hk == hv || throw(DimensionMismatch("k and v head counts must match"))
    skv == skvv || throw(DimensionMismatch("k and v sequence lengths must match"))
    bk == b && bv == b || throw(DimensionMismatch("q, k, and v batch sizes must match"))
    hq % hk == 0 || throw(DimensionMismatch("q heads must be a multiple of k/v heads"))
    size(out) == (d, hq, sq, b) ||
        throw(DimensionMismatch("out must have size $((d, hq, sq, b)), got $(size(out))"))
    return d, hq, sq, skv, b
end

function attention_stats_dims(stats, h, sq, b)
    stats === nothing && return
    size(stats) == (1, h, sq, b) ||
        throw(DimensionMismatch("stats must have size $((1, h, sq, b)), got $(size(stats))"))
    eltype(stats) == Float32 ||
        throw(ArgumentError("stats must be a Float32 DenseCuArray"))
    return
end

function attention_lengths_dims(seq_len_q, seq_len_kv, b)
    (seq_len_q === nothing) == (seq_len_kv === nothing) ||
        throw(ArgumentError("seq_len_q and seq_len_kv must be passed together"))
    seq_len_q === nothing && return
    size(seq_len_q) == (1, 1, 1, b) ||
        throw(DimensionMismatch("seq_len_q must have size $((1, 1, 1, b)), got $(size(seq_len_q))"))
    size(seq_len_kv) == (1, 1, 1, b) ||
        throw(DimensionMismatch("seq_len_kv must have size $((1, 1, 1, b)), got $(size(seq_len_kv))"))
    eltype(seq_len_q) == Int32 ||
        throw(ArgumentError("seq_len_q must be an Int32 DenseCuArray"))
    eltype(seq_len_kv) == Int32 ||
        throw(ArgumentError("seq_len_kv must be an Int32 DenseCuArray"))
    return
end

attention_optional_key(a) =
    a === nothing ? nothing : (eltype(a), size(a), strides(a), pointer_alignment(a))

# scale is a by-value binding supplied at execution time, so it does not key the plan
function attention_key(out, q, k, v, stats, seq_len_q, seq_len_kv, causal,
                        deterministic, math_mode, max_workspace)
    (:attention_fwd, eltype(q),
     size(q), strides(q), pointer_alignment(q),
     size(k), strides(k), pointer_alignment(k),
     size(v), strides(v), pointer_alignment(v),
     size(out), strides(out), pointer_alignment(out),
     attention_optional_key(stats),
     attention_optional_key(seq_len_q), attention_optional_key(seq_len_kv),
     causal, deterministic, math_mode, max_workspace)
end

function attention_backward_key(dq, dk, dv, dO, q, k, v, o, stats, seq_len_q,
                                 seq_len_kv, causal, deterministic, math_mode,
                                 max_workspace)
    (:attention_bwd, eltype(q),
     size(q), strides(q), pointer_alignment(q),
     size(k), strides(k), pointer_alignment(k),
     size(v), strides(v), pointer_alignment(v),
     size(o), strides(o), pointer_alignment(o),
     size(dO), strides(dO), pointer_alignment(dO),
     size(stats), strides(stats), pointer_alignment(stats),
     size(dq), strides(dq), pointer_alignment(dq),
     size(dk), strides(dk), pointer_alignment(dk),
     size(dv), strides(dv), pointer_alignment(dv),
     attention_optional_key(seq_len_q), attention_optional_key(seq_len_kv),
     causal, deterministic, math_mode, max_workspace)
end

function build_attention_graph(out, q, k, v, stats, seq_len_q, seq_len_kv, causal;
                                deterministic, math_mode, max_workspace)
    g = Graph(io_dtype=eltype(q), intermediate_dtype=Float32, compute_dtype=Float32)
    tq = tensor!(g, q; name="Q", backend_order=SDPA_BACKEND_ORDER)
    tk = tensor!(g, k; name="K", backend_order=SDPA_BACKEND_ORDER)
    tv = tensor!(g, v; name="V", backend_order=SDPA_BACKEND_ORDER)
    to = tensor!(g, out; name="O", output=true, backend_order=SDPA_BACKEND_ORDER)
    ts = scalar!(g, Float32; rank=4, name="Scale")
    tstats = stats === nothing ? nothing :
             tensor!(g, stats; name="Stats", output=true, backend_order=SDPA_BACKEND_ORDER)
    tseqq = seq_len_q === nothing ? nothing :
            tensor!(g, seq_len_q; name="SeqLenQ", backend_order=SDPA_BACKEND_ORDER)
    tseqkv = seq_len_kv === nothing ? nothing :
             tensor!(g, seq_len_kv; name="SeqLenKV", backend_order=SDPA_BACKEND_ORDER)
    sdpa_fwd!(g, tq, tk, tv; o=to, scale=ts, stats=tstats, seq_len_q=tseqq,
              seq_len_kv=tseqkv, causal)
    build!(g; deterministic, math_mode, max_workspace)
end

function build_attention_backward_graph(dq, dk, dv, dO, q, k, v, o, stats, seq_len_q,
                                         seq_len_kv, causal; deterministic, math_mode,
                                         max_workspace)
    g = Graph(io_dtype=eltype(q), intermediate_dtype=Float32, compute_dtype=Float32)
    tq = tensor!(g, q; name="Q", backend_order=SDPA_BACKEND_ORDER)
    tk = tensor!(g, k; name="K", backend_order=SDPA_BACKEND_ORDER)
    tv = tensor!(g, v; name="V", backend_order=SDPA_BACKEND_ORDER)
    to = tensor!(g, o; name="O", backend_order=SDPA_BACKEND_ORDER)
    tdO = tensor!(g, dO; name="dO", backend_order=SDPA_BACKEND_ORDER)
    tstats = tensor!(g, stats; name="Stats", backend_order=SDPA_BACKEND_ORDER)
    tdq = tensor!(g, dq; name="dQ", output=true, backend_order=SDPA_BACKEND_ORDER)
    tdk = tensor!(g, dk; name="dK", output=true, backend_order=SDPA_BACKEND_ORDER)
    tdv = tensor!(g, dv; name="dV", output=true, backend_order=SDPA_BACKEND_ORDER)
    ts = scalar!(g, Float32; rank=4, name="Scale")
    tseqq = seq_len_q === nothing ? nothing :
            tensor!(g, seq_len_q; name="SeqLenQ", backend_order=SDPA_BACKEND_ORDER)
    tseqkv = seq_len_kv === nothing ? nothing :
             tensor!(g, seq_len_kv; name="SeqLenKV", backend_order=SDPA_BACKEND_ORDER)
    sdpa_bwd!(g, tq, tk, tv, to, tdO, tstats; dQ=tdq, dK=tdk, dV=tdv, scale=ts,
              seq_len_q=tseqq, seq_len_kv=tseqkv, causal)
    build!(g; deterministic, math_mode, max_workspace)
end

function attention!(out::DenseCuArray{T,4}, q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                    v::DenseCuArray{T,4};
                    scale::Real=1/sqrt(size(q, 1)),
                    causal::Bool=false, dropout_p::Real=0, bias=nothing,
                    stats::Union{Nothing,DenseCuArray{Float32,4}}=nothing,
                    seq_len_q::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                    seq_len_kv::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                    deterministic::Bool=false,
                    math_mode=CUDACore.math_mode(),
                    max_workspace::Union{Nothing,Integer}=nothing) where {T}
    T in (Float16, BFloat16) ||
        throw(ArgumentError("cuDNN attention! only supports Float16/BFloat16, got $T"))
    dropout_p == 0 || throw(ArgumentError("cuDNN attention! dropout is not implemented yet"))
    bias === nothing || throw(ArgumentError("cuDNN attention! bias is not implemented yet"))
    isempty(out) && return out
    d, h, sq, skv, b = attention_dims(q, k, v, out)
    attention_stats_dims(stats, h, sq, b)
    attention_lengths_dims(seq_len_q, seq_len_kv, b)
    d % 8 == 0 || throw(ArgumentError("head dimension must be a multiple of 8, got $d"))
    d <= 256 || throw(ArgumentError("head dimension must be <= 256, got $d"))

    key = attention_key(out, q, k, v, stats, seq_len_q, seq_len_kv, causal,
                         deterministic, math_mode, max_workspace)
    g = cached_graph(key) do
        build_attention_graph(out, q, k, v, stats, seq_len_q, seq_len_kv, causal;
                               deterministic, math_mode, max_workspace)
    end

    bindings = IdDict{Tensor,Any}(
        tensor(g, "Q") => q,
        tensor(g, "K") => k,
        tensor(g, "V") => v,
        tensor(g, "O") => out,
        tensor(g, "Scale") => Float32(scale),
    )
    stats === nothing || (bindings[tensor(g, "Stats")] = stats)
    seq_len_q === nothing || (bindings[tensor(g, "SeqLenQ")] = seq_len_q)
    seq_len_kv === nothing || (bindings[tensor(g, "SeqLenKV")] = seq_len_kv)
    causal && (bindings[tensor(g, "MaskValue")] = Float32(-Inf))
    execute!(g, bindings)
    return out
end

function attention_backward!(dq::DenseCuArray{T,4}, dk::DenseCuArray{T,4},
                             dv::DenseCuArray{T,4}, dO::DenseCuArray{T,4},
                             q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                             v::DenseCuArray{T,4}, o::DenseCuArray{T,4},
                             stats::DenseCuArray{Float32,4};
                             scale::Real=1/sqrt(size(q, 1)),
                             causal::Bool=false, dropout_p::Real=0, bias=nothing,
                             seq_len_q::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                             seq_len_kv::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                             deterministic::Bool=false,
                             math_mode=CUDACore.math_mode(),
                             max_workspace::Union{Nothing,Integer}=nothing) where {T}
    T in (Float16, BFloat16) ||
        throw(ArgumentError("cuDNN attention_backward! only supports Float16/BFloat16, got $T"))
    dropout_p == 0 ||
        throw(ArgumentError("cuDNN attention_backward! dropout is not implemented yet"))
    bias === nothing || throw(ArgumentError("cuDNN attention_backward! bias is not implemented yet"))
    isempty(dq) && isempty(dk) && isempty(dv) && return dq, dk, dv
    d, h, sq, skv, b = attention_dims(q, k, v, o)
    size(dO) == size(q) ||
        throw(DimensionMismatch("dO must have size $(size(q)), got $(size(dO))"))
    size(dq) == size(q) ||
        throw(DimensionMismatch("dQ must have size $(size(q)), got $(size(dq))"))
    size(dk) == size(k) ||
        throw(DimensionMismatch("dK must have size $(size(k)), got $(size(dk))"))
    size(dv) == size(v) ||
        throw(DimensionMismatch("dV must have size $(size(v)), got $(size(dv))"))
    attention_stats_dims(stats, h, sq, b)
    attention_lengths_dims(seq_len_q, seq_len_kv, b)
    d % 8 == 0 || throw(ArgumentError("head dimension must be a multiple of 8, got $d"))
    d <= 256 || throw(ArgumentError("head dimension must be <= 256, got $d"))

    key = attention_backward_key(dq, dk, dv, dO, q, k, v, o, stats, seq_len_q,
                                  seq_len_kv, causal, deterministic, math_mode,
                                  max_workspace)
    g = cached_graph(key) do
        build_attention_backward_graph(dq, dk, dv, dO, q, k, v, o, stats, seq_len_q,
                                        seq_len_kv, causal; deterministic, math_mode,
                                        max_workspace)
    end

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
    )
    seq_len_q === nothing || (bindings[tensor(g, "SeqLenQ")] = seq_len_q)
    seq_len_kv === nothing || (bindings[tensor(g, "SeqLenKV")] = seq_len_kv)
    causal && (bindings[tensor(g, "MaskValue")] = Float32(-Inf))
    execute!(g, bindings)
    return dq, dk, dv
end

function attention_backward(dO::DenseCuArray{T,4}, q::DenseCuArray{T,4},
                            k::DenseCuArray{T,4}, v::DenseCuArray{T,4},
                            o::DenseCuArray{T,4}, stats::DenseCuArray{Float32,4};
                            kwargs...) where {T}
    dq, dk, dv = similar(q), similar(k), similar(v)
    attention_backward!(dq, dk, dv, dO, q, k, v, o, stats; kwargs...)
end

function attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4}, v::DenseCuArray{T,4};
                   kwargs...) where {T}
    out = similar(q)
    attention!(out, q, k, v; kwargs...)
end

function attention_supported(out::DenseCuArray{T,4}, q::DenseCuArray{T,4},
                             k::DenseCuArray{T,4}, v::DenseCuArray{T,4};
                             causal::Bool=false,
                             stats::Union{Nothing,DenseCuArray{Float32,4}}=nothing,
                             seq_len_q::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                             seq_len_kv::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                             deterministic::Bool=false,
                             math_mode=CUDACore.math_mode(),
                             max_workspace::Union{Nothing,Integer}=nothing) where {T}
    T in (Float16, BFloat16) || return false
    d = size(q, 1)
    (d % 8 == 0 && d <= 256) || return false
    isempty(out) && return true
    d, h, sq, skv, b = attention_dims(q, k, v, out)
    attention_stats_dims(stats, h, sq, b)
    attention_lengths_dims(seq_len_q, seq_len_kv, b)
    key = attention_key(out, q, k, v, stats, seq_len_q, seq_len_kv, causal,
                        deterministic, math_mode, max_workspace)
    try
        cached_graph(key) do
            build_attention_graph(out, q, k, v, stats, seq_len_q, seq_len_kv, causal;
                                  deterministic, math_mode, max_workspace)
        end
        return true
    catch e
        graph_unsupported(e) || rethrow()
        return false
    end
end

function attention_backward_supported(dq::DenseCuArray{T,4}, dk::DenseCuArray{T,4},
                                      dv::DenseCuArray{T,4}, dO::DenseCuArray{T,4},
                                      q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                                      v::DenseCuArray{T,4}, o::DenseCuArray{T,4},
                                      stats::DenseCuArray{Float32,4};
                                      causal::Bool=false,
                                      seq_len_q::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                                      seq_len_kv::Union{Nothing,DenseCuArray{Int32,4}}=nothing,
                                      deterministic::Bool=false,
                                      math_mode=CUDACore.math_mode(),
                                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    T in (Float16, BFloat16) || return false
    d = size(q, 1)
    (d % 8 == 0 && d <= 256) || return false
    isempty(dq) && isempty(dk) && isempty(dv) && return true
    d, h, sq, skv, b = attention_dims(q, k, v, o)
    attention_stats_dims(stats, h, sq, b)
    attention_lengths_dims(seq_len_q, seq_len_kv, b)
    key = attention_backward_key(dq, dk, dv, dO, q, k, v, o, stats, seq_len_q,
                                 seq_len_kv, causal, deterministic, math_mode,
                                 max_workspace)
    try
        cached_graph(key) do
            build_attention_backward_graph(dq, dk, dv, dO, q, k, v, o, stats, seq_len_q,
                                           seq_len_kv, causal; deterministic, math_mode,
                                           max_workspace)
        end
        return true
    catch e
        graph_unsupported(e) || rethrow()
        return false
    end
end

@doc raw"""
    attention(q, k, v; kwargs...)
    attention!(out, q, k, v; kwargs...)
    attention_backward(dO, q, k, v, o, stats; kwargs...)
    attention_backward!(dQ, dK, dV, dO, q, k, v, o, stats; kwargs...)
    attention_supported(out, q, k, v; kwargs...) -> Bool
    attention_backward_supported(dQ, dK, dV, dO, q, k, v, o, stats; kwargs...) -> Bool

Execute fused scaled dot-product attention with tensors shaped
`(head_dim, heads, sequence_length, batch_size)`.

Supported inputs are `Float16` and `BFloat16` `DenseCuArray`s. `scale` defaults to
`1 / sqrt(head_dim)`. Forward can write the Float32 `stats` tensor needed by backward.
Set `causal=true` for top-left causal forward masking. Dense padding masks are enabled by
passing Int32 `seq_len_q` and `seq_len_kv` tensors shaped `(1, 1, 1, batch_size)`.
`dropout_p` and `bias` are reserved for cuDNN graph features that are not wired yet.

The `_supported` predicates report whether cuDNN can execute the corresponding call,
building (and caching) the execution plan without running it. Callers with a generic
implementation can use them to decide between the fused and fallback paths; engine
coverage differs between forward and backward on some architectures.
"""
attention, attention!, attention_backward, attention_backward!, attention_supported,
attention_backward_supported

@public attention, attention!, attention_backward, attention_backward!,
        attention_supported, attention_backward_supported
