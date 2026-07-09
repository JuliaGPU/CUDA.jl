using CUDA

using cuDNN:
    conv_dgrad!,
    conv_fprop!,
    conv_wgrad!,
    Graph,
    execute!,
    is_supported,
    matmul!,
    resample_bwd!,
    resample_fwd!,
    tensor!

CUDA.allowscalar(false)

function matmul_ref(a, b)
    K, M, B = size(a)
    N = size(b, 1)
    out = Array{Float32}(undef, N, M, B)
    aa = Float32.(Array(a))
    bb = Float32.(Array(b))
    for batch in 1:B
        out[:, :, batch] = bb[:, :, batch] * aa[:, :, batch]
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
    a = CuArray(reshape(Float16.(sin.(1:K*M*B)), K, M, B))
    b = CuArray(reshape(Float16.(cos.(1:N*K*B)), N, K, B))
    y = CUDA.zeros(Float16, N, M, B)

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

let W=8, H=7, C=3, N=2, K=5
    x = CuArray(reshape(Float16.(sin.(1:W*H*C*N)), W, H, C, N) ./ 32)
    w = CuArray(reshape(Float16.(cos.(1:3*2*C*K)), 3, 2, C, K) ./ 32)
    pre_padding = (1, 0)
    post_padding = (2, 1)
    stride = (2, 1)
    dilation = (1, 2)
    ref = conv2d_ref(x, w; pre_padding, post_padding, stride, dilation)
    y = CUDA.zeros(Float16, size(ref)...)

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

    dx = CUDA.zeros(Float16, size(x))
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

    dw = CUDA.zeros(Float16, size(w))
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
    y = CUDA.zeros(Float32, size(ref))

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
    dx = CUDA.zeros(Float32, size(x))
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
