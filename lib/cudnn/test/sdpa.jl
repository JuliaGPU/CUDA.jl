using BFloat16s: BFloat16
using cuDNN: attention, attention!, attention_backward, attention_backward!, build!, execute!,
             Graph, scalar!, sdpa_fwd!, tensor!
using cuRAND

function seq_lens(seq_len, default, b)
    seq_len === nothing && return fill(default, b)
    return Int.(vec(Array(seq_len)))
end

function zero_padded_queries!(a, seq_len_q)
    seq_len_q === nothing && return a
    lens = seq_lens(seq_len_q, size(a, 3), size(a, 4))
    for bb in axes(a, 4)
        lens[bb] < size(a, 3) && (a[:, :, lens[bb]+1:end, bb] .= 0)
    end
    return a
end

# key j is visible to query i when i - left_bound < j <= i + right_bound, the causal
# mask being a right bound of 0, shifted by kvlen - qlen when bottom-right
function band_mask!(S, causal, left_bound, right_bound, qlen, kvlen)
    causal === false || (right_bound = something(right_bound, 0))
    shift = causal === :bottom_right ? kvlen - qlen : 0
    for i in axes(S, 1), j in axes(S, 2)
        right_bound !== nothing && j > i + shift + right_bound && (S[i, j] = Float32(-Inf))
        left_bound !== nothing && j <= i + shift - left_bound && (S[i, j] = Float32(-Inf))
    end
    return S
end

function sdpa_ref(q, k, v; scale, causal=false, left_bound=nothing, right_bound=nothing,
                  stats=false, seq_len_q=nothing, seq_len_kv=nothing)
    d, h, sq, b = size(q); hk = size(k, 2); skv = size(k, 3)
    groups = h ÷ hk
    qh, kh, vh = Float32.(Array(q)), Float32.(Array(k)), Float32.(Array(v))
    o = zeros(Float32, d, h, sq, b)
    s = stats ? zeros(Float32, 1, h, sq, b) : nothing
    qlens = seq_lens(seq_len_q, sq, b)
    kvlens = seq_lens(seq_len_kv, skv, b)
    for bb in 1:b, hh in 1:h
        kvh = (hh - 1) ÷ groups + 1
        Q, K, V = qh[:, hh, :, bb], kh[:, kvh, :, bb], vh[:, kvh, :, bb]
        S = scale .* (Q' * K)                       # sq x skv
        band_mask!(S, causal, left_bound, right_bound, qlens[bb], kvlens[bb])
        kvlens[bb] < skv && (S[:, kvlens[bb]+1:end] .= Float32(-Inf))
        m = maximum(S; dims=2)
        e = exp.(S .- m)
        l = sum(e; dims=2)
        A = e ./ l                                  # sq x skv
        qlens[bb] < sq && (A[qlens[bb]+1:end, :] .= 0)
        o[:, hh, :, bb] = V * A'                    # d x sq
        stats && (s[1, hh, :, bb] .= vec(m .+ log.(l)))
    end
    return stats ? (o, s) : o
end

function sdpa_bwd_ref(q, k, v, dO; scale, causal=false, left_bound=nothing,
                      right_bound=nothing, seq_len_q=nothing, seq_len_kv=nothing)
    d, h, sq, b = size(q); hk = size(k, 2); skv = size(k, 3)
    groups = h ÷ hk
    qh = Float32.(Array(q))
    kh = Float32.(Array(k))
    vh = Float32.(Array(v))
    dOh = Float32.(Array(dO))
    dq = zeros(Float32, size(q))
    dk = zeros(Float32, size(k))
    dv = zeros(Float32, size(v))
    qlens = seq_lens(seq_len_q, sq, b)
    kvlens = seq_lens(seq_len_kv, skv, b)
    for bb in 1:b, hh in 1:h
        kvh = (hh - 1) ÷ groups + 1
        Q, K, V = qh[:, hh, :, bb], kh[:, kvh, :, bb], vh[:, kvh, :, bb]
        S = scale .* (Q' * K)                       # sq x skv
        band_mask!(S, causal, left_bound, right_bound, qlens[bb], kvlens[bb])
        kvlens[bb] < skv && (S[:, kvlens[bb]+1:end] .= Float32(-Inf))
        m = maximum(S; dims=2)
        A = exp.(S .- m)
        A ./= sum(A; dims=2)
        qlens[bb] < sq && (A[qlens[bb]+1:end, :] .= 0)
        dOhh = dOh[:, hh, :, bb]
        dv[:, kvh, :, bb] .+= dOhh * A
        dA = dOhh' * V
        dS = A .* (dA .- sum(dA .* A; dims=2))
        dq[:, hh, :, bb] .= scale .* (K * dS')
        dk[:, kvh, :, bb] .+= scale .* (Q * dS)
    end
    return dq, dk, dv
end

function sdpatest(T; d=64, sq=32, skv=32, h=4, hk=h, b=2, scale=1/sqrt(d), rtol=2e-2,
                  causal=false, left_bound=nothing, right_bound=nothing)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    band = (; causal, left_bound, right_bound)

    ref = sdpa_ref(q, k, v; scale, band...)
    y = attention(q, k, v; scale, band...)
    @test size(y) == (d, h, sq, b)
    @test eltype(y) == T
    @test Array(y) ≈ ref rtol=rtol

    out = similar(q)
    attention!(out, q, k, v; scale, band...)
    @test Array(out) == Array(y)
end

function sdpa_stats_test(T; d=64, sq=32, skv=32, h=4, hk=h, b=2,
                         scale=1/sqrt(d), causal=false, left_bound=nothing,
                         right_bound=nothing, rtol=2e-3)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    band = (; causal, left_bound, right_bound)

    ref, refstats = sdpa_ref(q, k, v; scale, band..., stats=true)
    out = similar(q)
    stats = CUDACore.zeros(Float32, 1, sq, h, b)
    attention!(out, q, k, v; scale, band..., stats)
    @test Array(out) ≈ ref rtol=2e-2
    @test Array(stats) ≈ permutedims(refstats, (1, 3, 2, 4)) rtol=rtol
end

function sdpa_backward_test(T; d=64, sq=32, skv=32, h=4, hk=h, b=2,
                            scale=1/sqrt(d), causal=false, left_bound=nothing,
                            right_bound=nothing, deterministic=false, rtol=3e-2)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    dO = cuRAND.randn(T, d, h, sq, b) ./ 4
    band = (; causal, left_bound, right_bound)
    o = similar(q)
    stats = CUDACore.zeros(Float32, 1, sq, h, b)
    attention!(o, q, k, v; scale, band..., stats)

    refdq, refdk, refdv = sdpa_bwd_ref(q, k, v, dO; scale, band...)
    dq, dk, dv = similar(q), similar(k), similar(v)
    if !cuDNN.attention_backward_supported(dq, dk, dv, dO, q, k, v, o, stats; band...,
                                           deterministic)
        @test_skip "SDPA backward engine is unsupported on this device"
        return
    end
    attention_backward!(dq, dk, dv, dO, q, k, v, o, stats; scale, band..., deterministic)
    @test Array(dq) ≈ refdq rtol=rtol
    @test Array(dk) ≈ refdk rtol=rtol
    @test Array(dv) ≈ refdv rtol=rtol

    # dQ is accumulated in arbitrary order unless deterministic, so repeated calls
    # otherwise agree only closely
    agree(a, b) = deterministic ? a == b : isapprox(a, b; rtol=1e-2)
    a_dq, a_dk, a_dv = attention_backward(dO, q, k, v, o, stats; scale, band...,
                                          deterministic)
    @test agree(Array(a_dq), Array(dq))
    @test agree(Array(a_dk), Array(dk))
    @test agree(Array(a_dv), Array(dv))
end

function sdpa_padding_test(T; d=64, sq=64, skv=64, h=4, hk=2, b=2,
                           qlens=[32, 48], kvlens=[40, 56], causal=false,
                           left_bound=nothing, right_bound=nothing,
                           deterministic=false, scale=1/sqrt(d), rtol=3e-2)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    seq_len_q = CuArray(reshape(Int32.(qlens), 1, 1, 1, b))
    seq_len_kv = CuArray(reshape(Int32.(kvlens), 1, 1, 1, b))
    band = (; causal, left_bound, right_bound)

    ref = sdpa_ref(q, k, v; scale, band..., seq_len_q, seq_len_kv)
    out = similar(q)
    stats = CUDACore.zeros(Float32, 1, sq, h, b)
    if !cuDNN.attention_supported(out, q, k, v; band..., stats, seq_len_q, seq_len_kv)
        @test_skip "SDPA sequence-length engine is unsupported on this device"
        return
    end
    attention!(out, q, k, v; scale, band..., stats, seq_len_q, seq_len_kv)
    got = zero_padded_queries!(Array(out), seq_len_q)
    expected = zero_padded_queries!(copy(ref), seq_len_q)
    @test got ≈ expected rtol=rtol

    # padded backward: gradients compare inside the valid region; out-of-range
    # rows are clamped away by the GEMM overrides and stay zero
    dO = cuRAND.randn(T, d, h, sq, b) ./ 4
    refdq, refdk, refdv = sdpa_bwd_ref(q, k, v, dO; scale, band..., seq_len_q,
                                       seq_len_kv)
    dq = CUDACore.zeros(T, d, h, sq, b)
    dk = CUDACore.zeros(T, d, hk, skv, b)
    dv = CUDACore.zeros(T, d, hk, skv, b)
    if !cuDNN.attention_backward_supported(dq, dk, dv, dO, q, k, v, out, stats;
                                           band..., seq_len_q, seq_len_kv, deterministic)
        @test_skip "SDPA padded backward engine is unsupported on this device"
        return
    end
    attention_backward!(dq, dk, dv, dO, q, k, v, out, stats; scale, band...,
                        seq_len_q, seq_len_kv, deterministic)
    @test zero_padded_queries!(Array(dq), seq_len_q) ≈
          zero_padded_queries!(refdq, seq_len_q) rtol=rtol
    @test zero_padded_queries!(Array(dk), seq_len_kv) ≈
          zero_padded_queries!(refdk, seq_len_kv) rtol=rtol
    @test zero_padded_queries!(Array(dv), seq_len_kv) ≈
          zero_padded_queries!(refdv, seq_len_kv) rtol=rtol
end

# the valid prefix of each batch element, back to back along the token dimension
pack(a, lens) = cat((a[:, :, 1:n, bb] for (bb, n) in enumerate(lens))...; dims=3)

function sdpa_packed_test(T; d=64, h=4, hk=h, qlens=[5, 17, 64], kvlens=[9, 17, 70],
                          causal=false, left_bound=nothing, right_bound=nothing,
                          slack=0, scale=1/sqrt(d), rtol=3e-2)
    b = length(qlens)
    sq, skv = maximum(qlens) + slack, maximum(kvlens) + slack
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    dO = cuRAND.randn(T, d, h, sq, b) ./ 4
    seq_len_q = CuArray(reshape(Int32.(qlens), 1, 1, 1, b))
    seq_len_kv = CuArray(reshape(Int32.(kvlens), 1, 1, 1, b))
    band = (; causal, left_bound, right_bound)
    ref, refstats = sdpa_ref(q, k, v; scale, band..., stats=true, seq_len_q, seq_len_kv)
    refdq, refdk, refdv = sdpa_bwd_ref(q, k, v, dO; scale, band..., seq_len_q, seq_len_kv)

    pq, pdO = CuArray(pack(Array(q), qlens)), CuArray(pack(Array(dO), qlens))
    pk, pv = CuArray(pack(Array(k), kvlens)), CuArray(pack(Array(v), kvlens))
    kw = (; cu_seq_len_q=CuArray(Int32[0; cumsum(qlens)]),
          cu_seq_len_kv=CuArray(Int32[0; cumsum(kvlens)]),
          max_seq_len_q=sq, max_seq_len_kv=skv, band...)
    out = similar(pq)
    stats = CUDACore.zeros(Float32, 1, sq, h, b)
    if !cuDNN.attention_supported(out, pq, pk, pv; stats, kw...)
        @test_skip "packed SDPA is unsupported on this device"
        return
    end
    attention!(out, pq, pk, pv; scale, stats, kw...)
    @test Array(out) ≈ pack(ref, qlens) rtol=rtol
    valid(s) = reduce(vcat, vec(s[:, 1:n, :, bb]) for (bb, n) in enumerate(qlens))
    @test valid(Array(stats)) ≈ valid(permutedims(refstats, (1, 3, 2, 4))) rtol=2e-3
    @test Array(attention(pq, pk, pv; scale, kw...)) == Array(out)

    dq, dk, dv = similar(pq), similar(pk), similar(pv)
    if !cuDNN.attention_backward_supported(dq, dk, dv, pdO, pq, pk, pv, out, stats; kw...)
        @test_skip "packed SDPA backward is unsupported on this device"
        return
    end
    attention_backward!(dq, dk, dv, pdO, pq, pk, pv, out, stats; scale, kw...)
    @test Array(dq) ≈ pack(refdq, qlens) rtol=rtol
    @test Array(dk) ≈ pack(refdk, kvlens) rtol=rtol
    @test Array(dv) ≈ pack(refdv, kvlens) rtol=rtol
    a_dq, a_dk, a_dv = attention_backward(pdO, pq, pk, pv, out, stats; scale, kw...)
    @test Array(a_dq) ≈ Array(dq) rtol=1e-2
    @test Array(a_dk) ≈ Array(dk) rtol=1e-2
    @test Array(a_dv) ≈ Array(dv) rtol=1e-2
end

if capability(device()) >= v"8.0"
    for T in (Float16, BFloat16)
        sdpatest(T)                                  # default square case
        sdpatest(T; sq=128, skv=128)                 # longer sequences
        sdpatest(T; d=32, sq=16, skv=48, h=2, b=3)   # non-square sq != skv
        sdpatest(T; scale=0.5)                        # custom scale
        sdpatest(T; d=128, sq=64, skv=64, h=8)       # larger head dim
        sdpatest(T; h=4, hk=2)                        # grouped-query attention
        sdpatest(T; causal=true)                      # top-left causal mask
        sdpatest(T; h=4, hk=2, causal=true)           # causal GQA
        sdpa_stats_test(T)
        sdpa_stats_test(T; h=4, hk=2, causal=true)
        sdpa_backward_test(T)
        sdpa_backward_test(T; h=4, hk=2)
        sdpa_backward_test(T; causal=true)
        sdpa_backward_test(T; h=4, hk=2, causal=true)
        sdpa_padding_test(T)

        # bottom-right causal mask: the queries are the last positions of the keys
        sdpatest(T; causal=:bottom_right)                       # square: as top-left
        sdpatest(T; sq=16, skv=48, causal=:bottom_right)        # chunked prefill
        sdpatest(T; sq=1, skv=40, h=4, hk=2, causal=:bottom_right)  # GQA decoding
        sdpa_stats_test(T; sq=16, skv=48, causal=:bottom_right)
        sdpa_backward_test(T; sq=16, skv=48, causal=:bottom_right)
        sdpa_backward_test(T; sq=24, skv=100, h=4, hk=2, causal=:bottom_right)
        sdpa_padding_test(T; causal=:bottom_right)
        # the diagonal shifts by each batch element's own lengths, not the sizes'
        sdpa_padding_test(T; sq=16, skv=64, qlens=[4, 16], kvlens=[20, 64],
                          causal=:bottom_right)

        # deterministic backward: dQ without the out-of-order accumulator
        sdpa_backward_test(T; deterministic=true)
        sdpa_backward_test(T; sq=24, skv=100, h=4, hk=2, causal=:bottom_right,
                           deterministic=true)
        sdpa_padding_test(T; deterministic=true)

        # packed sequences
        sdpa_packed_test(T)
        sdpa_packed_test(T; causal=true)
        sdpa_packed_test(T; h=8, hk=2, causal=:bottom_right)   # per-sequence diagonals
        sdpa_packed_test(T; d=128, hk=1, qlens=[100, 3, 30, 1],
                         kvlens=[100, 50, 30, 200])            # multi-query attention
        sdpa_packed_test(T; qlens=[1, 1, 1], kvlens=[30, 100, 7],
                         causal=:bottom_right)                 # decoding against caches
        sdpa_packed_test(T; qlens=[5, 0, 64], kvlens=[9, 0, 70],
                         slack=13)                             # empty sequence, loose max

        # diagonal bands
        sdpatest(T; sq=128, skv=128, causal=true, left_bound=16)          # sliding window
        sdpatest(T; h=4, hk=2, causal=true, left_bound=8)                 # with GQA
        sdpatest(T; sq=16, skv=48, causal=:bottom_right, left_bound=20)   # chunked prefill
        sdpatest(T; sq=1, skv=40, causal=:bottom_right, left_bound=10)    # decoding
        sdpatest(T; left_bound=5, right_bound=4)                          # |i - j| < 5
        sdpatest(T; left_bound=6)
        sdpatest(T; right_bound=3)
        sdpatest(T; sq=16, skv=48, causal=:bottom_right, left_bound=7, right_bound=2)
        sdpa_stats_test(T; causal=true, left_bound=8)
        sdpa_backward_test(T; causal=true, left_bound=8)
        sdpa_backward_test(T; sq=24, skv=100, h=4, hk=2, causal=:bottom_right,
                           left_bound=30)
        sdpa_backward_test(T; left_bound=5, right_bound=4)
        sdpa_backward_test(T; causal=true, left_bound=8, deterministic=true)
        sdpa_padding_test(T; causal=true, left_bound=16)
        sdpa_padding_test(T; sq=16, skv=64, qlens=[4, 16], kvlens=[20, 64],
                          causal=:bottom_right, left_bound=10)
        sdpa_packed_test(T; causal=true, left_bound=8)
        sdpa_packed_test(T; h=8, hk=2, causal=:bottom_right, left_bound=12)
    end

    # long enough for the default plan's dQ to vary between runs
    let q = cuRAND.randn(Float16, 64, 4, 512, 2) ./ 4
        dO = cuRAND.randn(Float16, size(q)) ./ 4
        stats = CUDACore.zeros(Float32, 1, 512, 4, 2)
        o = attention(q, q, q; stats)
        if cuDNN.attention_backward_supported(similar(q), similar(q), similar(q), dO, q, q,
                                              q, o, stats; deterministic=true)
            runs = [Array.(attention_backward(dO, q, q, q, o, stats; deterministic=true))
                    for _ in 1:3]
            @test all(==(runs[1]), runs)
        else
            @test_skip "deterministic SDPA backward is unsupported on this device"
        end
    end

    let q = cuRAND.randn(Float16, 64, 4, 30) ./ 4
        cu = CuArray(Int32[0, 10, 30])
        kw = (; cu_seq_len_q=cu, cu_seq_len_kv=cu, max_seq_len_q=20, max_seq_len_kv=20)
        @test_throws DimensionMismatch attention(q, q, q; kw...,
                                                 cu_seq_len_kv=CuArray(Int32[0, 30]))
        @test_throws ArgumentError attention(q, q, q; kw..., max_seq_len_q=0)
        @test_throws UndefKeywordError attention(q, q, q; cu_seq_len_q=cu, cu_seq_len_kv=cu)
        stats_bad = CUDACore.zeros(Float32, 1, 30, 4, 2)    # padded to max_seq_len_q
        @test_throws DimensionMismatch attention(q, q, q; kw..., stats=stats_bad)

        # the token count does not shape the plan
        stats = CUDACore.zeros(Float32, 1, 20, 4, 2)
        o = attention(q, q, q; kw..., stats)
        attention_backward(q, q, q, q, o, stats; kw...)
        n = length(cuDNN.handle().plans)
        q40 = cuRAND.randn(Float16, 64, 4, 40) ./ 4
        kw40 = (; kw..., cu_seq_len_q=CuArray(Int32[0, 20, 40]),
                cu_seq_len_kv=CuArray(Int32[0, 20, 40]))
        o40 = attention(q40, q40, q40; kw40..., stats)
        attention_backward(q40, q40, q40, q40, o40, stats; kw40...)
        @test length(cuDNN.handle().plans) == n

        # cudnn-frontend refuses deterministic packed backward on these GPUs
        if capability(device()).major in (8, 12)
            @test !cuDNN.attention_backward_supported(similar(q), similar(q), similar(q), q,
                                                      q, q, q, o, stats; kw...,
                                                      deterministic=true)
            @test_throws cuDNN.UnsupportedGraphError attention_backward(
                q, q, q, q, o, stats; kw..., deterministic=true)
        end
    end

    let q = CuArray(Float16.(randn(Float32, 64, 4, 64, 2)))
        qview = view(q, :, :, 1:32, :)  # non-contiguous: dense strides would be wrong
        @test_throws MethodError attention(qview, qview, qview)
        qhost = Array(q)                # host memory must not reach cuDNN as a device pointer
        @test_throws MethodError attention(qhost, qhost, qhost)
        qf32 = Float32.(q)              # unsupported by the fused engine
        @test_throws ArgumentError attention(qf32, qf32, qf32)

        k_bad = CuArray(Float16.(randn(Float32, 32, 4, 64, 2)))
        @test_throws DimensionMismatch attention(q, k_bad, q)
        v_bad = CuArray(Float16.(randn(Float32, 64, 4, 63, 2)))
        @test_throws DimensionMismatch attention(q, q, v_bad)
        out_bad = CuArray{Float16}(undef, 64, 4, 63, 2)
        @test_throws DimensionMismatch attention!(out_bad, q, q, q)
        q60 = CuArray(Float16.(randn(Float32, 60, 4, 32, 2)))
        @test_throws ArgumentError attention(q60, q60, q60)

        out = similar(q)
        seq_len_q = CuArray(reshape(Int32[16, 32], 1, 1, 1, 2))
        seq_len_kv_bad = CuArray(reshape(Int32[16, 32, 32], 1, 1, 1, 3))
        @test_throws ArgumentError attention!(out, q, q, q; seq_len_q)
        @test_throws DimensionMismatch attention!(out, q, q, q; seq_len_q,
                                                  seq_len_kv=seq_len_kv_bad)
    end

    let d=64, h=4, sq=32, skv=32, b=2, scale=1f0/8, dev=device()
        q = cuRAND.randn(Float16, d, h, sq, b) ./ 4
        k = cuRAND.randn(Float16, d, h, skv, b) ./ 4
        v = cuRAND.randn(Float16, d, h, skv, b) ./ 4
        ref = sdpa_ref(q, k, v; scale)
        tasks = [Threads.@spawn begin
                     device!(dev)
                     attention(q, k, v; scale)
                 end for _ in 1:8]
        for t in tasks
            @test Array(fetch(t)) ≈ ref rtol=2e-2
        end
    end

    let q = cuRAND.randn(Float16, 64, 4, 32, 2) ./ 4
        attention(q, q, q)
        n = length(cuDNN.handle().plans)
        attention(q, q, q)
        @test length(cuDNN.handle().plans) == n
    end

    let q = cuRAND.randn(Float16, 64, 4, 32, 2) ./ 4
        out = similar(q)
        @test cuDNN.attention_supported(out, q, q, q)
        @test cuDNN.attention_supported(out, q, q, q; causal=true)
        @test cuDNN.attention_supported(out, q, q, q; causal=:bottom_right)
        k16 = cuRAND.randn(Float16, 64, 4, 16, 2) ./ 4   # more queries than keys
        @test !cuDNN.attention_supported(out, q, k16, k16; causal=:bottom_right)
        @test_throws cuDNN.UnsupportedGraphError attention(q, k16, k16;
                                                           causal=:bottom_right)
        @test_throws ArgumentError attention(q, q, q; causal=:diagonal)
        @test_throws ArgumentError cuDNN.attention_supported(out, q, q, q; causal=:diagonal)
        q60 = cuRAND.randn(Float16, 60, 4, 32, 2)
        @test !cuDNN.attention_supported(similar(q60), q60, q60, q60)
        q0 = CuArray{Float16}(undef, 60, 4, 0, 2)    # empty: nothing to reject
        @test isempty(attention(q0, q0, q0))
        qf32 = cuRAND.randn(Float32, 64, 4, 32, 2)
        @test !cuDNN.attention_supported(similar(qf32), qf32, qf32, qf32)
    end

    let q = cuRAND.randn(Float16, 64, 4, 32, 2) ./ 4
        # the bounds' values don't shape the plan, and a right bound of 0 is causal
        attention(q, q, q; causal=true)
        attention(q, q, q; causal=true, left_bound=4)
        n = length(cuDNN.handle().plans)
        attention(q, q, q; causal=true, left_bound=9)
        attention(q, q, q; right_bound=0)
        @test length(cuDNN.handle().plans) == n
        @test Array(attention(q, q, q; right_bound=0)) ==
              Array(attention(q, q, q; causal=true))

        @test_throws ArgumentError attention(q, q, q; left_bound=0)
        @test_throws ArgumentError attention(q, q, q; right_bound=-1)
        k16 = cuRAND.randn(Float16, 64, 4, 16, 2) ./ 4   # more queries than keys
        @test !cuDNN.attention_supported(similar(q), q, k16, k16; causal=true, left_bound=4)
        @test_throws cuDNN.UnsupportedGraphError attention(q, k16, k16; causal=true,
                                                           left_bound=4)
    end

    let d=64, h=4, s=32, b=2, scale=1f0/8
        q = cuRAND.randn(Float16, d, h, s, b) ./ 4
        k = cuRAND.randn(Float16, d, h, s, b) ./ 4
        v = cuRAND.randn(Float16, d, h, s, b) ./ 4
        out = similar(q)
        g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
        tq = tensor!(g, q; name="Q")
        tk = tensor!(g, k; name="K")
        tv = tensor!(g, v; name="V")
        to = tensor!(g, out; name="O", output=true)
        ts = scalar!(g, Float32; rank=4, name="Scale")
        sdpa_fwd!(g, tq, tk, tv; o=to, scale=ts)
        build!(g)
        execute!(g, tq=>q, tk=>k, tv=>v, to=>out, ts=>scale)
        @test Array(out) ≈ sdpa_ref(q, k, v; scale) rtol=2e-2
    end
else
    @warn "Skipping SDPA tests: fused attention requires compute capability >= 8.0"
end
