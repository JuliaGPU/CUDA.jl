using CUDA
using cuDNN: convolution!, convolution_data_gradient!, convolution_filter_gradient!

CUDA.allowscalar(false)

conv_ref_type(::Type{Float64}) = Float64
conv_ref_type(::Type) = Float32

function conv2d_ref(x, w; pre_padding, post_padding, stride, dilation)
    wx, hx, cin, n = size(x)
    ww, hw, cfilter, cout = size(w)
    cin == cfilter || throw(DimensionMismatch("grouped convolution not supported here"))
    outw = fld(wx + pre_padding[1] + post_padding[1] - dilation[1] * (ww - 1) - 1,
               stride[1]) + 1
    outh = fld(hx + pre_padding[2] + post_padding[2] - dilation[2] * (hw - 1) - 1,
               stride[2]) + 1
    R = conv_ref_type(eltype(x))
    xx = R.(Array(x))
    wwgt = R.(Array(w))
    y = zeros(R, outw, outh, cout, n)
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
    R = conv_ref_type(eltype(dy))
    ddy = R.(Array(dy))
    wwgt = R.(Array(w))
    dx = zeros(R, x_size)
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
    R = conv_ref_type(eltype(dy))
    ddy = R.(Array(dy))
    xx = R.(Array(x))
    dw = zeros(R, w_size)
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

function conv1d_wgrad_ref(dy, x, w_size; pre_padding, stride, dilation)
    ww, cin, cout = w_size
    wx, cx, n = size(x)
    cin == cx || throw(DimensionMismatch("grouped convolution not supported here"))
    R = conv_ref_type(eltype(dy))
    ddy = R.(Array(dy))
    xx = R.(Array(x))
    dw = zeros(R, w_size)
    for batch in 1:n, co in 1:cout, ox in 1:size(dy, 1)
        grad = ddy[ox, co, batch]
        for ci in 1:cin, kx in 1:ww
            ix = (ox - 1) * stride[1] + (kx - 1) * dilation[1] - pre_padding[1] + 1
            if 1 <= ix <= wx
                dw[kx, ci, co] += grad * xx[ix, ci, batch]
            end
        end
    end
    return dw
end

let W=8, H=7, C=3, N=2, K=5
    x = CuArray(reshape(Float16.(sin.(1:W*H*C*N)), W, H, C, N) ./ 32)
    w = CuArray(reshape(Float16.(cos.(1:3*2*C*K)), 3, 2, C, K) ./ 32)
    y0 = CuArray(reshape(Float16.(sin.(1:5*6*K*N)), 5, 6, K, N) ./ 64)
    y = copy(y0)
    pre_padding = (1, 0)
    post_padding = (2, 1)
    stride = (2, 1)
    dilation = (1, 2)
    alpha = 1.5f0
    beta = 0.25f0
    ref = alpha .* conv2d_ref(x, w; pre_padding, post_padding, stride, dilation) .+
          beta .* Float32.(Array(y0))

    supported = true
    try
        convolution!(y, x, w; padding=(1, 2, 0, 1), stride, dilation, alpha, beta)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        @test_skip "convolution! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Float32.(Array(y)) ≈ ref rtol=3f-2 atol=3f-2
    end

    dy = CuArray(reshape(Float16.(sin.(1:length(ref))), size(ref)) ./ 64)
    dx0 = CuArray(reshape(Float16.(cos.(1:W*H*C*N)), W, H, C, N) ./ 128)
    dx = copy(dx0)
    dgrad_ref = alpha .* conv2d_dgrad_ref(dy, w, size(x); pre_padding, stride, dilation) .+
                 beta .* Float32.(Array(dx0))
    supported = true
    try
        convolution_data_gradient!(dx, dy, w; padding=(1, 2, 0, 1), stride, dilation,
                                   alpha, beta)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        @test_skip "convolution_data_gradient! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Float32.(Array(dx)) ≈ dgrad_ref rtol=3f-2 atol=3f-2
    end

    dw0 = CuArray(reshape(Float16.(cos.(1:length(w))), size(w)) ./ 128)
    dw = copy(dw0)
    wgrad_ref = alpha .* conv2d_wgrad_ref(dy, x, size(w); pre_padding, stride, dilation) .+
                 beta .* Float32.(Array(dw0))
    supported = true
    try
        convolution_filter_gradient!(dw, x, dy; padding=(1, 2, 0, 1), stride, dilation,
                                     alpha, beta)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        @test_skip "convolution_filter_gradient! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Float32.(Array(dw)) ≈ wgrad_ref rtol=3f-2 atol=3f-2
    end
end

let W=8, H=7, C=3, N=2, K=5
    x = CuArray(reshape(Float16.(sin.(1:W*H*C*N)), W, H, C, N) ./ 32)
    w = CuArray(reshape(Float16.(cos.(1:3*2*C*K)), 3, 2, C, K) ./ 32)
    z = CuArray(reshape(Float16.(cos.(1:5*6*K*N)), 5, 6, K, N) ./ 64)
    bias = CuArray(reshape(Float16.(sin.(1:K)), 1, 1, K, 1) ./ 16)
    y = similar(z)
    pre_padding = (1, 0)
    post_padding = (2, 1)
    stride = (2, 1)
    dilation = (1, 2)
    alpha = 1.25f0
    beta = 0.5f0
    ref = alpha .* conv2d_ref(x, w; pre_padding, post_padding, stride, dilation) .+
          beta .* Float32.(Array(z)) .+ Float32.(Array(bias))
    ref = max.(ref, 0f0)

    convolution!(y, x, w; padding=(1, 2, 0, 1), stride, dilation, alpha, beta,
                 z, bias, activation=:relu)
    @test Float32.(Array(y)) ≈ ref rtol=3f-2 atol=3f-2
end

let W=4, H=4, C=1, N=1, K=1
    x = CuArray(reshape(Float64.(1:W*H*C*N), W, H, C, N))
    w = CuArray(reshape(Float64.(1:4), 2, 2, C, K))
    y0 = CuArray(reshape(Float64.(sin.(1:9)), 3, 3, K, N))
    y = copy(y0)
    alpha = 1.25
    beta = 0.5
    ref = alpha .* Float64.(conv2d_ref(x, w; pre_padding=(0, 0),
                                       post_padding=(0, 0), stride=(1, 1),
                                       dilation=(1, 1))) .+ beta .* Array(y0)
    convolution!(y, x, w; alpha, beta, compute_type=Float64)
    @test Array(y) ≈ ref rtol=1e-12 atol=1e-12

    dy = CuArray(reshape(Float64.(cos.(1:9)), size(ref)))
    dx0 = CuArray(reshape(Float64.(sin.(1:W*H*C*N)), W, H, C, N))
    dx = copy(dx0)
    dref = alpha .* Float64.(conv2d_dgrad_ref(dy, w, size(x); pre_padding=(0, 0),
                                              stride=(1, 1), dilation=(1, 1))) .+
           beta .* Array(dx0)
    convolution_data_gradient!(dx, dy, w; alpha, beta, compute_type=Float64)
    @test Array(dx) ≈ dref rtol=1e-12 atol=1e-12

    dw0 = CuArray(reshape(Float64.(sin.(1:4)), size(w)))
    dw = copy(dw0)
    wref = alpha .* Float64.(conv2d_wgrad_ref(dy, x, size(w); pre_padding=(0, 0),
                                              stride=(1, 1), dilation=(1, 1))) .+
           beta .* Array(dw0)
    convolution_filter_gradient!(dw, x, dy; alpha, beta, compute_type=Float64)
    @test Array(dw) ≈ wref rtol=1e-12 atol=1e-12
end

let W=5, C=2, N=1, K=3
    x = CuArray(reshape(Float64.(1:W*C*N), W, C, N))
    alpha = 1.25
    beta = 0.5
    stride = (1,)
    dilation = (1,)
    for (pre, post) in ((1, 0), (0, 1), (2, 3))
        ylen = fld(W + pre + post - 2, 1) + 1
        dy = CuArray(reshape(Float64.(1:ylen*K*N), ylen, K, N))
        dw0 = CuArray(reshape(Float64.(sin.(1:2*C*K)), 2, C, K))
        dw = copy(dw0)
        ref = alpha .* conv1d_wgrad_ref(dy, x, size(dw); pre_padding=(pre,),
                                        stride, dilation) .+ beta .* Array(dw0)
        convolution_filter_gradient!(dw, x, dy; padding=(pre, post), stride, dilation,
                                     alpha, beta, compute_type=Float64)
        @test Array(dw) ≈ ref rtol=1e-12 atol=1e-12
    end
end
