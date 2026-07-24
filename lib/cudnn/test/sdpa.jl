using BFloat16s: BFloat16
import CUDA
import cuDNN
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

function sdpa_ref(q, k, v; scale, causal=false, stats=false, seq_len_q=nothing,
                  seq_len_kv=nothing)
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
        if causal
            for i in 1:sq, j in i+1:skv
                S[i, j] = Float32(-Inf)
            end
        end
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

function sdpa_bwd_ref(q, k, v, dO; scale, causal=false, seq_len_q=nothing,
                      seq_len_kv=nothing)
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
        if causal
            for i in 1:sq, j in i+1:skv
                S[i, j] = Float32(-Inf)
            end
        end
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
                  causal=false)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4

    ref = sdpa_ref(q, k, v; scale, causal)
    y = attention(q, k, v; scale, causal)
    @test size(y) == (d, h, sq, b)
    @test eltype(y) == T
    @test Array(y) ≈ ref rtol=rtol

    out = similar(q)
    attention!(out, q, k, v; scale, causal)
    @test Array(out) == Array(y)
end

function sdpa_stats_test(T; d=64, sq=32, skv=32, h=4, hk=h, b=2,
                         scale=1/sqrt(d), causal=false, rtol=2e-3)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4

    ref, refstats = sdpa_ref(q, k, v; scale, causal, stats=true)
    out = similar(q)
    stats = CUDA.zeros(Float32, 1, h, sq, b)
    attention!(out, q, k, v; scale, causal, stats)
    @test Array(out) ≈ ref rtol=2e-2
    @test Array(stats) ≈ refstats rtol=rtol
end

function sdpa_backward_test(T; d=64, sq=32, skv=32, h=4, hk=h, b=2,
                            scale=1/sqrt(d), causal=false, rtol=3e-2)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    dO = cuRAND.randn(T, d, h, sq, b) ./ 4
    o = similar(q)
    stats = CUDA.zeros(Float32, 1, h, sq, b)
    attention!(o, q, k, v; scale, causal, stats)

    refdq, refdk, refdv = sdpa_bwd_ref(q, k, v, dO; scale, causal)
    dq, dk, dv = similar(q), similar(k), similar(v)
    if !cuDNN.attention_backward_supported(dq, dk, dv, dO, q, k, v, o, stats; causal)
        @test_skip "SDPA backward engine is unsupported on this device"
        return
    end
    attention_backward!(dq, dk, dv, dO, q, k, v, o, stats; scale, causal)
    @test Array(dq) ≈ refdq rtol=rtol
    @test Array(dk) ≈ refdk rtol=rtol
    @test Array(dv) ≈ refdv rtol=rtol

    a_dq, a_dk, a_dv = attention_backward(dO, q, k, v, o, stats; scale, causal)
    @test Array(a_dq) == Array(dq)
    @test Array(a_dk) == Array(dk)
    @test Array(a_dv) == Array(dv)
end

function sdpa_padding_test(T; d=64, sq=64, skv=64, h=4, hk=2, b=2,
                           scale=1/sqrt(d), rtol=3e-2)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, hk, skv, b) ./ 4
    v = cuRAND.randn(T, d, hk, skv, b) ./ 4
    seq_len_q = CuArray(reshape(Int32[32, 48], 1, 1, 1, b))
    seq_len_kv = CuArray(reshape(Int32[40, 56], 1, 1, 1, b))

    ref = sdpa_ref(q, k, v; scale, seq_len_q, seq_len_kv)
    out = similar(q)
    stats = CUDA.zeros(Float32, 1, h, sq, b)
    if !cuDNN.attention_supported(out, q, k, v; stats, seq_len_q, seq_len_kv)
        @test_skip "SDPA sequence-length engine is unsupported on this device"
        return
    end
    attention!(out, q, k, v; scale, stats, seq_len_q, seq_len_kv)
    got = zero_padded_queries!(Array(out), seq_len_q)
    expected = zero_padded_queries!(copy(ref), seq_len_q)
    @test got ≈ expected rtol=rtol

    dO = cuRAND.randn(T, d, h, sq, b) ./ 4
    refdq, refdk, refdv = sdpa_bwd_ref(q, k, v, dO; scale, seq_len_q, seq_len_kv)
    dq, dk, dv = similar(q), similar(k), similar(v)
    if !cuDNN.attention_backward_supported(dq, dk, dv, dO, q, k, v, out, stats;
                                           seq_len_q, seq_len_kv)
        @test_skip "SDPA sequence-length backward engine is unsupported on this device"
        return
    end
    attention_backward!(dq, dk, dv, dO, q, k, v, out, stats; scale, seq_len_q,
                        seq_len_kv)
    gotdq = zero_padded_queries!(Array(dq), seq_len_q)
    refdq = zero_padded_queries!(refdq, seq_len_q)
    @test gotdq ≈ refdq rtol=rtol
    @test Array(dk) ≈ refdk rtol=rtol
    @test Array(dv) ≈ refdv rtol=rtol
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
        sdpa_padding_test(T)
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
        q60 = cuRAND.randn(Float16, 60, 4, 32, 2)
        @test !cuDNN.attention_supported(similar(q60), q60, q60, q60)
        qf32 = cuRAND.randn(Float32, 64, 4, 32, 2)
        @test !cuDNN.attention_supported(similar(qf32), qf32, qf32, qf32)
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
