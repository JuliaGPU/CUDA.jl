using cuDNN: cudnnSDPAForward, cudnnSDPAForward!

# Reference dense attention in Float32 for the (d, h, s, b) layout (matches NNlib's layout).
function sdpa_ref(q, k, v; scale)
    d, h, sq, b = size(q); skv = size(k, 3)
    qh, kh, vh = Float32.(Array(q)), Float32.(Array(k)), Float32.(Array(v))
    o = zeros(Float32, d, h, sq, b)
    for bb in 1:b, hh in 1:h
        Q, K, V = qh[:, hh, :, bb], kh[:, hh, :, bb], vh[:, hh, :, bb]
        S = scale .* (Q' * K)                       # sq × skv
        m = maximum(S; dims=2)
        e = exp.(S .- m)
        A = e ./ sum(e; dims=2)                     # sq × skv
        o[:, hh, :, bb] = V * A'                    # d × sq
    end
    return o
end

function sdpatest(T; d=64, sq=32, skv=32, h=4, b=2, scale=1/sqrt(d), rtol=2e-2)
    q = CuArray(T.(randn(Float32, d, h, sq, b) ./ 4))
    k = CuArray(T.(randn(Float32, d, h, skv, b) ./ 4))
    v = CuArray(T.(randn(Float32, d, h, skv, b) ./ 4))

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

# NOTE: only Float16 is exercised here. On Blackwell (sm_120) + cuDNN 9.20, BFloat16 fused SDPA
# hangs in cuDNN's execute (a cuDNN/driver issue, not in this wrapper); the BFloat16 code path
# is identical and works on other architectures.
if capability(device()) >= v"8.0"
    T = Float16
    sdpatest(T)                                  # default square case
    sdpatest(T; sq=128, skv=128)                 # longer sequences
    sdpatest(T; d=32, sq=16, skv=48, h=2, b=3)   # non-square sq != skv
    sdpatest(T; scale=0.5)                        # custom scale
    sdpatest(T; d=128, sq=64, skv=64, h=8)       # larger head dim
else
    @warn "Skipping SDPA tests: fused attention requires compute capability >= 8.0"
end
