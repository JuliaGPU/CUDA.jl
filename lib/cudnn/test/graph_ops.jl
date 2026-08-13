using cuDNN:
    block_scale_dequantize!,
    block_scale_quantize!,
    conv_dgrad!,
    conv_fprop!,
    conv_wgrad!,
    Graph,
    execute!,
    is_supported,
    matmul!,
    norm_bwd!,
    norm_fwd!,
    output!,
    pointwise!,
    resample_bwd!,
    resample_fwd!,
    tensor,
    tensor!,
    CUDNN_DATA_FP8_E4M3,
    CUDNN_DATA_FP8_E8M0,
    CUDNN_TENSOR_REORDERING_F8_128x4

CUDACore.allowscalar(false)

function matmul_ref(a, b)
    M, K, B = size(a)
    N = size(b, 2)
    out = Array{Float32}(undef, M, N, B)
    aa = Float32.(Array(a))
    bb = Float32.(Array(b))
    for batch in 1:B
        out[:, :, batch] = aa[:, :, batch] * bb[:, :, batch]
    end
    return out
end

function conv2d_ref(x, w; pre_padding, post_padding, stride, dilation)
    wx, hx, cin, n = size(x)
    ww, hw, cfilter, cout = size(w)
    cin == cfilter || throw(DimensionMismatch("grouped convolution not supported here"))
    outw = fld(wx + pre_padding[1] + post_padding[1] - dilation[1] * (ww - 1) - 1,
               stride[1]) + 1
    outh = fld(hx + pre_padding[2] + post_padding[2] - dilation[2] * (hw - 1) - 1,
               stride[2]) + 1
    xx = Float32.(Array(x))
    wwgt = Float32.(Array(w))
    y = zeros(Float32, outw, outh, cout, n)
    for batch in 1:n, co in 1:cout, oy in 1:outh, ox in 1:outw
        acc = 0f0
        for ci in 1:cin, ky in 1:hw, kx in 1:ww
            ix = (ox - 1) * stride[1] + (kx - 1) * dilation[1] - pre_padding[1] + 1
            iy = (oy - 1) * stride[2] + (ky - 1) * dilation[2] - pre_padding[2] + 1
            if 1 <= ix <= wx && 1 <= iy <= hx
                acc += xx[ix, iy, ci, batch] * wwgt[kx, ky, ci, co]
            end
        end
        y[ox, oy, co, batch] = acc
    end
    return y
end

function conv2d_dgrad_ref(dy, w, x_size; pre_padding, stride, dilation)
    wx, hx, cin, n = x_size
    ww, hw, cfilter, cout = size(w)
    cin == cfilter || throw(DimensionMismatch("grouped convolution not supported here"))
    ddy = Float32.(Array(dy))
    wwgt = Float32.(Array(w))
    dx = zeros(Float32, x_size)
    for batch in 1:n, co in 1:cout, oy in 1:size(dy, 2), ox in 1:size(dy, 1)
        grad = ddy[ox, oy, co, batch]
        for ci in 1:cin, ky in 1:hw, kx in 1:ww
            ix = (ox - 1) * stride[1] + (kx - 1) * dilation[1] - pre_padding[1] + 1
            iy = (oy - 1) * stride[2] + (ky - 1) * dilation[2] - pre_padding[2] + 1
            if 1 <= ix <= wx && 1 <= iy <= hx
                dx[ix, iy, ci, batch] += grad * wwgt[kx, ky, ci, co]
            end
        end
    end
    return dx
end

function conv2d_wgrad_ref(dy, x, w_size; pre_padding, stride, dilation)
    ww, hw, cin, cout = w_size
    wx, hx, cx, n = size(x)
    cin == cx || throw(DimensionMismatch("grouped convolution not supported here"))
    ddy = Float32.(Array(dy))
    xx = Float32.(Array(x))
    dw = zeros(Float32, w_size)
    for batch in 1:n, co in 1:cout, oy in 1:size(dy, 2), ox in 1:size(dy, 1)
        grad = ddy[ox, oy, co, batch]
        for ci in 1:cin, ky in 1:hw, kx in 1:ww
            ix = (ox - 1) * stride[1] + (kx - 1) * dilation[1] - pre_padding[1] + 1
            iy = (oy - 1) * stride[2] + (ky - 1) * dilation[2] - pre_padding[2] + 1
            if 1 <= ix <= wx && 1 <= iy <= hx
                dw[kx, ky, ci, co] += grad * xx[ix, iy, ci, batch]
            end
        end
    end
    return dw
end

function avgpool2d_ref(x; window, pre_padding, stride, include_pad::Bool)
    wx, hx, c, n = size(x)
    outw = fld(wx + 2 * pre_padding[1] - window[1], stride[1]) + 1
    outh = fld(hx + 2 * pre_padding[2] - window[2], stride[2]) + 1
    xx = Float32.(Array(x))
    y = zeros(Float32, outw, outh, c, n)
    for batch in 1:n, ch in 1:c, oy in 1:outh, ox in 1:outw
        acc = 0f0
        valid = 0
        for ky in 1:window[2], kx in 1:window[1]
            ix = (ox - 1) * stride[1] + kx - pre_padding[1]
            iy = (oy - 1) * stride[2] + ky - pre_padding[2]
            if 1 <= ix <= wx && 1 <= iy <= hx
                acc += xx[ix, iy, ch, batch]
                valid += 1
            end
        end
        denom = include_pad ? window[1] * window[2] : valid
        y[ox, oy, ch, batch] = acc / denom
    end
    return y
end

function avgpool2d_bwd_ref(dy, x_size; window, pre_padding, stride, include_pad::Bool)
    wx, hx, c, n = x_size
    ddy = Float32.(Array(dy))
    dx = zeros(Float32, x_size)
    for batch in 1:n, ch in 1:c, oy in 1:size(dy, 2), ox in 1:size(dy, 1)
        valid = 0
        for ky in 1:window[2], kx in 1:window[1]
            ix = (ox - 1) * stride[1] + kx - pre_padding[1]
            iy = (oy - 1) * stride[2] + ky - pre_padding[2]
            valid += (1 <= ix <= wx && 1 <= iy <= hx)
        end
        denom = include_pad ? window[1] * window[2] : valid
        grad = ddy[ox, oy, ch, batch] / denom
        for ky in 1:window[2], kx in 1:window[1]
            ix = (ox - 1) * stride[1] + kx - pre_padding[1]
            iy = (oy - 1) * stride[2] + ky - pre_padding[2]
            if 1 <= ix <= wx && 1 <= iy <= hx
                dx[ix, iy, ch, batch] += grad
            end
        end
    end
    return dx
end

let K=16, M=16, N=16, B=2
    a = CuArray(reshape(Float16.(sin.(1:M*K*B)), M, K, B))
    b = CuArray(reshape(Float16.(cos.(1:K*N*B)), K, N, B))
    y = CUDACore.zeros(Float16, M, N, B)

    g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    ta = tensor!(g, a; name="A")
    tb = tensor!(g, b; name="B")
    ty = tensor!(g, y; name="Y", output=true)
    matmul!(g, ta, tb; c=ty)

    if is_supported(g)
        execute!(g, ta=>a, tb=>b, ty=>y)
        @test Float32.(Array(y)) ≈ matmul_ref(a, b) rtol=2f-3 atol=2f-3
    else
        @test_skip is_supported(g)
    end
end

# binary pointwise: alpha2 must default to 1 — the scaling constants mean
# y = op(alpha1·x, alpha2·b), so a zero default drops the second operand
let M=16, N=16
    x = CuArray(reshape(Float16.(sin.(1:M*N)), M, N, 1))
    b = CuArray(reshape(Float16.(cos.(1:M*N)), M, N, 1))
    y = CUDACore.zeros(Float16, M, N, 1)

    g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    tx = tensor!(g, x; name="X")
    tb = tensor!(g, b; name="B")
    ty = tensor!(g, y; name="Y", output=true)
    pointwise!(g, :add, tx, tb; y=ty)

    if is_supported(g)
        execute!(g, tx=>x, tb=>b, ty=>y)
        @test Float32.(Array(y)) ≈ Float32.(Array(x)) .+ Float32.(Array(b)) rtol=1f-3 atol=1f-3
    else
        @test_skip is_supported(g)
    end
end

let W=8, H=7, C=3, N=2, K=5
    x = CuArray(reshape(Float16.(sin.(1:W*H*C*N)), W, H, C, N) ./ 32)
    w = CuArray(reshape(Float16.(cos.(1:3*2*C*K)), 3, 2, C, K) ./ 32)
    pre_padding = (1, 0)
    post_padding = (2, 1)
    stride = (2, 1)
    dilation = (1, 2)
    ref = conv2d_ref(x, w; pre_padding, post_padding, stride, dilation)
    y = CUDACore.zeros(Float16, size(ref)...)

    g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    tx = tensor!(g, x; name="X")
    tw = tensor!(g, w; name="W")
    ty = tensor!(g, y; name="Y", output=true)
    conv_fprop!(g, tx, tw; y=ty, pre_padding, post_padding, stride, dilation)

    if is_supported(g)
        execute!(g, tx=>x, tw=>w, ty=>y)
        @test Float32.(Array(y)) ≈ ref rtol=3f-2 atol=3f-2
    else
        @test_skip is_supported(g)
    end

    dy = CuArray(reshape(Float16.(sin.(1:length(ref))), size(ref)) ./ 64)

    dx = CUDACore.zeros(Float16, size(x))
    dgrad_ref = conv2d_dgrad_ref(dy, w, size(x); pre_padding, stride, dilation)
    gd = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    tdy = tensor!(gd, dy; name="dY")
    twd = tensor!(gd, w; name="W")
    tdx = tensor!(gd, dx; name="dX", output=true)
    conv_dgrad!(gd, tdy, twd; dx=tdx, pre_padding, post_padding, stride, dilation)
    if is_supported(gd)
        execute!(gd, tdy=>dy, twd=>w, tdx=>dx)
        @test Float32.(Array(dx)) ≈ dgrad_ref rtol=3f-2 atol=3f-2
    else
        @test_skip is_supported(gd)
    end

    dw = CUDACore.zeros(Float16, size(w))
    wgrad_ref = conv2d_wgrad_ref(dy, x, size(w); pre_padding, stride, dilation)
    gw = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    tyw = tensor!(gw, dy; name="dY")
    txw = tensor!(gw, x; name="X")
    tdw = tensor!(gw, dw; name="dW", output=true)
    conv_wgrad!(gw, tyw, txw; dw=tdw, pre_padding, post_padding, stride, dilation)
    if is_supported(gw)
        execute!(gw, tyw=>dy, txw=>x, tdw=>dw)
        @test Float32.(Array(dw)) ≈ wgrad_ref rtol=3f-2 atol=3f-2
    else
        @test_skip is_supported(gw)
    end
end

let W=7, H=6, C=2, N=2
    x = CuArray(reshape(Float32.(1:W*H*C*N), W, H, C, N) ./ 32)
    window = (2, 3)
    padding = (1, 1)
    stride = (2, 2)
    ref = avgpool2d_ref(x; window, pre_padding=padding, stride, include_pad=true)
    y = CUDACore.zeros(Float32, size(ref))

    g = Graph(io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32)
    tx = tensor!(g, x; name="X")
    ty = tensor!(g, y; name="Y", output=true)
    resample_fwd!(g, tx; y=ty, mode=:avgpool_include_padding, window,
                  pre_padding=padding, stride)
    if is_supported(g)
        execute!(g, tx=>x, ty=>y)
        @test Array(y) ≈ ref rtol=1f-5 atol=1f-5
    else
        @test_skip is_supported(g)
    end

    dy = CuArray(reshape(Float32.(sin.(1:length(ref))), size(ref)))
    dx = CUDACore.zeros(Float32, size(x))
    bwd_ref = avgpool2d_bwd_ref(dy, size(x); window, pre_padding=padding, stride,
                                include_pad=true)
    gb = Graph(io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32)
    txb = tensor!(gb, x; name="X")
    tyb = tensor!(gb, y; name="Y")
    tdy = tensor!(gb, dy; name="dY")
    tdx = tensor!(gb, dx; name="dX", output=true)
    resample_bwd!(gb, tdy; dx=tdx, x=txb, y=tyb, mode=:avgpool_include_padding, window,
                  pre_padding=padding, stride)
    if is_supported(gb)
        execute!(gb, txb=>x, tyb=>y, tdy=>dy, tdx=>dx)
        @test Array(dx) ≈ bwd_ref rtol=1f-5 atol=1f-5
    else
        @test_skip is_supported(gb)
    end
end

# block-scale requires Blackwell
blockscale_claimed = CUDACore.capability(CUDACore.device()) >= v"10.0"

# helper: MXFP8 operand pair, dequantized and multiplied. F8_128x4 scale
# dimensions are padded to whole 128×4 tiles
function blockscale_matmul!(g; K, M, N)
    MP, KS = cld(M, 128) * 128, cld(cld(K, 32), 4) * 4
    NP = cld(N, 128) * 128
    a = tensor!(g; dims=(M, K, 1), strides=(K, 1, M * K), dtype=CUDNN_DATA_FP8_E4M3,
                name="A")
    as = tensor!(g; dims=(MP, KS, 1), strides=(KS, 1, MP * KS),
                 dtype=CUDNN_DATA_FP8_E8M0, name="A.scale",
                 reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    da = block_scale_dequantize!(g, a, as; block_size=32)
    b = tensor!(g; dims=(K, N, 1), dtype=CUDNN_DATA_FP8_E4M3, name="B")
    bs = tensor!(g; dims=(KS, NP, 1), dtype=CUDNN_DATA_FP8_E8M0, name="B.scale",
                 reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    db = block_scale_dequantize!(g, b, bs; block_size=32)
    return matmul!(g, da, db)
end

let K=128, M=128, N=128
    g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
    output!(blockscale_matmul!(g; K, M, N))
    @test is_supported(g) == blockscale_claimed
end

let K=128, M=128, N=64
    g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
    output!(blockscale_matmul!(g; K, M, N))
    @test is_supported(g) == blockscale_claimed

    g2 = Graph()
    b = tensor!(g2; dims=(K, N, 1), dtype=CUDNN_DATA_FP8_E4M3)
    unpadded = tensor!(g2; dims=(K ÷ 32, N, 1), dtype=CUDNN_DATA_FP8_E8M0,
                       reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    @test_throws DimensionMismatch block_scale_dequantize!(g2, b, unpadded; block_size=32)
end

let K=160, M=128, N=128
    g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
    output!(blockscale_matmul!(g; K, M, N))
    @test is_supported(g) == blockscale_claimed
end

let K=128, M=128, N=128
    g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
    c = blockscale_matmul!(g; K, M, N)
    y, scale = block_scale_quantize!(g, c; block_size=32, block_dim=1,
                                     dtype=CUDNN_DATA_FP8_E4M3,
                                     scale_dtype=CUDNN_DATA_FP8_E8M0)
    @test y.dims == [M, N, 1]
    @test scale.dims == [M ÷ 32, N, 1]
    @test scale.reordering == CUDNN_TENSOR_REORDERING_F8_128x4
    @test is_supported(g) == blockscale_claimed
end

let
    g = Graph()
    x = tensor!(g; dims=(64, 200, 1), dtype=Float32)
    y, scale = block_scale_quantize!(g, x; block_size=32, block_dim=1,
                                     dtype=CUDNN_DATA_FP8_E4M3,
                                     scale_dtype=CUDNN_DATA_FP8_E8M0)
    @test scale.dims == [4, 256, 1]     # cld(64, 32) = 2 → 4; 200 → 256
    @test scale.strides == [1, 4, 1024]
end

let
    g = Graph()
    x = tensor!(g; dims=(128, 64, 1), dtype=CUDNN_DATA_FP8_E4M3)
    bad_scale = tensor!(g; dims=(128, 3, 1), dtype=CUDNN_DATA_FP8_E8M0)
    @test_throws DimensionMismatch block_scale_dequantize!(g, x, bad_scale; block_size=32)
    bad_rank = tensor!(g; dims=(128, 2), dtype=CUDNN_DATA_FP8_E8M0)
    @test_throws DimensionMismatch block_scale_dequantize!(g, x, bad_rank; block_size=32)
    unblocked = tensor!(g; dims=(128, 64, 1), dtype=CUDNN_DATA_FP8_E8M0)
    @test_throws DimensionMismatch block_scale_dequantize!(g, x, unblocked; block_size=32)
    @test_throws ArgumentError block_scale_dequantize!(g, x, bad_scale; block_size=1)
    hi = tensor!(g; dims=(128, 64, 1), dtype=Float32)
    @test_throws ArgumentError block_scale_quantize!(g, hi; block_size=32)
    @test_throws DimensionMismatch block_scale_quantize!(g, hi; block_size=32, block_dim=3,
                                                         dtype=CUDNN_DATA_FP8_E4M3,
                                                         scale_dtype=CUDNN_DATA_FP8_E8M0)
end

# layer and RMS norm normalize over the dimensions the scale spans; the
# statistics span the complement and, at inference, are computed on the fly
function layernorm_ref(x, scale, bias; epsilon)
    mean = sum(x; dims=1) ./ size(x, 1)
    var = sum(abs2, x .- mean; dims=1) ./ size(x, 1)
    return @. scale * (x - mean) / sqrt(var + epsilon) + bias
end

function rmsnorm_ref(x, scale; epsilon)
    ms = sum(abs2, x; dims=1) ./ size(x, 1)
    return @. scale * x / sqrt(ms + epsilon)
end

let H=64, S=8, B=2
    epsilon = 1f-4
    x_ref = reshape(Float32.(sin.(1:H*S*B)), H, S, B)
    scale_ref = reshape(1f0 .+ Float32.(cos.(1:H)) ./ 2, H, 1, 1)
    bias_ref = reshape(Float32.(sin.(1:H)) ./ 4, H, 1, 1)
    x, scale, bias = CuArray(x_ref), CuArray(scale_ref), CuArray(bias_ref)

    y = CUDACore.zeros(Float32, H, S, B)
    g = Graph()
    tx = tensor!(g, x; name="X")
    tscale = tensor!(g, scale; name="Scale")
    tbias = tensor!(g, bias; name="Bias")
    ty = tensor!(g, y; name="Y", output=true)
    norm_fwd!(g, tx, tscale, tbias; y=ty, mode=:layernorm, phase=:inference)
    if is_supported(g)
        execute!(g, tx=>x, tscale=>scale, tbias=>bias, ty=>y,
                 tensor(g, "Epsilon")=>epsilon)
        @test Array(y) ≈ layernorm_ref(x_ref, scale_ref, bias_ref; epsilon) rtol=1f-4 atol=1f-4
    else
        @test_skip is_supported(g)
    end

    yr = CUDACore.zeros(Float32, H, S, B)
    gr = Graph()
    rx = tensor!(gr, x; name="X")
    rscale = tensor!(gr, scale; name="Scale")
    ry = tensor!(gr, yr; name="Y", output=true)
    norm_fwd!(gr, rx, rscale, nothing; y=ry, mode=:rmsnorm, phase=:inference)
    if is_supported(gr)
        execute!(gr, rx=>x, rscale=>scale, ry=>yr, tensor(gr, "Epsilon")=>epsilon)
        @test Array(yr) ≈ rmsnorm_ref(x_ref, scale_ref; epsilon) rtol=1f-4 atol=1f-4
    else
        @test_skip is_supported(gr)
    end
end

# backward references, normalized over dimension 1 with statistics saved by
# the training forward
function layernorm_bwd_ref(dy, x, scale; epsilon)
    H = size(x, 1)
    mean = sum(x; dims=1) ./ H
    var = sum(abs2, x .- mean; dims=1) ./ H
    iv = @. 1 / sqrt(var + epsilon)
    xhat = @. (x - mean) * iv
    dxhat = @. dy * scale
    dx = iv .* (dxhat .- sum(dxhat; dims=1) ./ H .-
                xhat .* (sum(dxhat .* xhat; dims=1) ./ H))
    return dx, sum(dy .* xhat; dims=(2, 3)), sum(dy; dims=(2, 3))
end

function rmsnorm_bwd_ref(dy, x, scale; epsilon)
    H = size(x, 1)
    iv = @. 1 / sqrt($(sum(abs2, x; dims=1)) / H + epsilon)
    xhat = @. x * iv
    dxhat = @. dy * scale
    dx = iv .* (dxhat .- xhat .* (sum(dxhat .* xhat; dims=1) ./ H))
    return dx, sum(dy .* xhat; dims=(2, 3))
end

# training forward to backward round trip: the forward saves the per-sample
# statistics, the backward consumes them; returns nothing when unsupported
function norm_roundtrip(mode, x, scale, bias, dy; epsilon)
    (H, S, B) = size(x)
    y, dx = CUDACore.zeros(Float32, H, S, B), CUDACore.zeros(Float32, H, S, B)
    sinv = CUDACore.zeros(Float32, 1, S, B)
    smean = mode === :rmsnorm ? nothing : CUDACore.zeros(Float32, 1, S, B)
    dscale = CUDACore.zeros(Float32, H, 1, 1)
    dbias = mode === :rmsnorm ? nothing : CUDACore.zeros(Float32, H, 1, 1)
    inout(g, a; kws...) = a === nothing ? nothing : tensor!(g, a; kws...)
    bind!(binds, t, a) = t === nothing ? binds : push!(binds, t => a)

    g = Graph()
    tx, tscale, tbias = tensor!(g, x; name="X"), tensor!(g, scale; name="S"),
                        inout(g, bias; name="B")
    ty = tensor!(g, y; name="Y", output=true)
    tinv = tensor!(g, sinv; name="IV", output=true)
    tmean = inout(g, smean; name="M", output=true)
    norm_fwd!(g, tx, tscale, tbias; y=ty, mean=tmean, inv_variance=tinv,
              mode, phase=:training)
    is_supported(g) || return nothing
    binds = Any[tx => x, tscale => scale, ty => y, tinv => sinv,
                tensor(g, "Epsilon") => epsilon]
    bind!(bind!(binds, tbias, bias), tmean, smean)
    execute!(g, binds...)

    gb = Graph()
    bdy, bx, bscale = tensor!(gb, dy; name="dY"), tensor!(gb, x; name="X"),
                      tensor!(gb, scale; name="S")
    binv, bmean = tensor!(gb, sinv; name="IV"), inout(gb, smean; name="M")
    bdx = tensor!(gb, dx; name="dX", output=true)
    bdscale = tensor!(gb, dscale; name="dS", output=true)
    bdbias = inout(gb, dbias; name="dB", output=true)
    norm_bwd!(gb, bdy, bx, bscale, bmean, binv;
              dx=bdx, dscale=bdscale, dbias=bdbias, mode)
    is_supported(gb) || return nothing
    binds = Any[bdy => dy, bx => x, bscale => scale, binv => sinv,
                bdx => dx, bdscale => dscale]
    bind!(bind!(binds, bmean, smean), bdbias, dbias)
    execute!(gb, binds...)
    return y, dx, dscale, dbias
end

let H=64, S=8, B=2
    epsilon = 1f-4
    x_ref = reshape(Float32.(sin.(1:H*S*B)), H, S, B)
    scale_ref = reshape(1f0 .+ Float32.(cos.(1:H)) ./ 2, H, 1, 1)
    bias_ref = reshape(Float32.(sin.(1:H)) ./ 4, H, 1, 1)
    dy_ref = reshape(Float32.(cos.(1:H*S*B)), H, S, B) ./ 2
    x, scale, bias, dy = CuArray.((x_ref, scale_ref, bias_ref, dy_ref))

    r = norm_roundtrip(:layernorm, x, scale, bias, dy; epsilon)
    if r === nothing
        @test_skip false
    else
        y, dx, dscale, dbias = r
        dx_ref, dscale_ref, dbias_ref = layernorm_bwd_ref(dy_ref, x_ref, scale_ref; epsilon)
        @test Array(y) ≈ layernorm_ref(x_ref, scale_ref, bias_ref; epsilon) rtol=1f-3 atol=1f-3
        @test Array(dx) ≈ dx_ref rtol=1f-3 atol=1f-3
        @test Array(dscale) ≈ dscale_ref rtol=1f-3 atol=1f-3
        @test Array(dbias) ≈ dbias_ref rtol=1f-3 atol=1f-3
    end

    r = norm_roundtrip(:rmsnorm, x, scale, nothing, dy; epsilon)
    if r === nothing
        @test_skip false
    else
        y, dx, dscale, dbias = r
        dx_ref, dscale_ref = rmsnorm_bwd_ref(dy_ref, x_ref, scale_ref; epsilon)
        @test dbias === nothing
        @test Array(y) ≈ rmsnorm_ref(x_ref, scale_ref; epsilon) rtol=1f-3 atol=1f-3
        @test Array(dx) ≈ dx_ref rtol=1f-3 atol=1f-3
        @test Array(dscale) ≈ dscale_ref rtol=1f-3 atol=1f-3
    end
end
