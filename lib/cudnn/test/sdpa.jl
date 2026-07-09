using BFloat16s: BFloat16
import cuDNN
using cuDNN: cudnnSDPAForward, cudnnSDPAForward!
using cuRAND

# Reference dense attention in Float32 for the (d, h, s, b) layout (matches NNlib's layout).
function sdpa_ref(q, k, v; scale)
    d, h, sq, b = size(q); skv = size(k, 3)
    qh, kh, vh = Float32.(Array(q)), Float32.(Array(k)), Float32.(Array(v))
    o = zeros(Float32, d, h, sq, b)
    for bb in 1:b, hh in 1:h
        Q, K, V = qh[:, hh, :, bb], kh[:, hh, :, bb], vh[:, hh, :, bb]
        S = scale .* (Q' * K)                       # sq x skv
        m = maximum(S; dims=2)
        e = exp.(S .- m)
        A = e ./ sum(e; dims=2)                     # sq x skv
        o[:, hh, :, bb] = V * A'                    # d x sq
    end
    return o
end

function sdpatest(T; d=64, sq=32, skv=32, h=4, b=2, scale=1/sqrt(d), rtol=2e-2)
    q = cuRAND.randn(T, d, h, sq, b) ./ 4
    k = cuRAND.randn(T, d, h, skv, b) ./ 4
    v = cuRAND.randn(T, d, h, skv, b) ./ 4

    ref = sdpa_ref(q, k, v; scale)
    y = cudnnSDPAForward(q, k, v; scale)
    @test size(y) == (d, h, sq, b)
    @test eltype(y) == T
    @test Array(y) ≈ ref rtol=rtol

    # in-place form must agree with the allocating form
    out = similar(q)
    cudnnSDPAForward!(out, q, k, v; scale)
    @test Array(out) == Array(y)
end

if capability(device()) >= v"8.0"
    for T in (Float16, BFloat16)
        sdpatest(T)                                  # default square case
        sdpatest(T; sq=128, skv=128)                 # longer sequences
        sdpatest(T; d=32, sq=16, skv=48, h=2, b=3)   # non-square sq != skv
        sdpatest(T; scale=0.5)                        # custom scale
        sdpatest(T; d=128, sq=64, skv=64, h=8)       # larger head dim
    end

    # invalid inputs are rejected rather than silently producing wrong results
    let q = CuArray(Float16.(randn(Float32, 64, 4, 64, 2)))
        qview = view(q, :, :, 1:32, :)  # non-contiguous: dense strides would be wrong
        @test_throws MethodError cudnnSDPAForward(qview, qview, qview)
        qhost = Array(q)                # host memory must not reach cuDNN as a device pointer
        @test_throws MethodError cudnnSDPAForward(qhost, qhost, qhost)
        qf32 = Float32.(q)              # unsupported by the fused engine
        @test_throws ArgumentError cudnnSDPAForward(qf32, qf32, qf32)
        @test_throws ArgumentError cudnnSDPAForward(q, q, q; is_causal=true)

        k_bad = CuArray(Float16.(randn(Float32, 32, 4, 64, 2)))
        @test_throws DimensionMismatch cudnnSDPAForward(q, k_bad, q)
        v_bad = CuArray(Float16.(randn(Float32, 64, 4, 63, 2)))
        @test_throws DimensionMismatch cudnnSDPAForward(q, q, v_bad)
        out_bad = CuArray{Float16}(undef, 64, 4, 63, 2)
        @test_throws DimensionMismatch cudnnSDPAForward!(out_bad, q, q, q)
        q60 = CuArray(Float16.(randn(Float32, 60, 4, 32, 2)))
        @test_throws ArgumentError cudnnSDPAForward(q60, q60, q60)
    end

    let d=64, h=4, sq=32, skv=32, b=2, scale=1f0/8, dev=device()
        q = cuRAND.randn(Float16, d, h, sq, b) ./ 4
        k = cuRAND.randn(Float16, d, h, skv, b) ./ 4
        v = cuRAND.randn(Float16, d, h, skv, b) ./ 4
        ref = sdpa_ref(q, k, v; scale)
        tasks = [Threads.@spawn begin
                     device!(dev)
                     cudnnSDPAForward(q, k, v; scale)
                 end for _ in 1:8]
        for t in tasks
            @test Array(fetch(t)) ≈ ref rtol=2e-2
        end
    end

    let q = cuRAND.randn(Float16, 64, 4, 32, 2) ./ 4
        cudnnSDPAForward(q, q, q)
        n = length(cuDNN.handle().plans)
        cudnnSDPAForward(q, q, q)
        @test length(cuDNN.handle().plans) == n
    end
else
    @warn "Skipping SDPA tests: fused attention requires compute capability >= 8.0"
end
