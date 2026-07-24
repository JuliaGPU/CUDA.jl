using CUDA
using cuDNN: maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, UnsupportedGraphError

CUDA.allowscalar(false)

function pool_output_dims(x_size; window, pre_padding, post_padding, stride)
    (fld(x_size[1] + pre_padding[1] + post_padding[1] - window[1], stride[1]) + 1,
     fld(x_size[2] + pre_padding[2] + post_padding[2] - window[2], stride[2]) + 1)
end

function maxpool2d_ref(x; window, pre_padding, post_padding, stride)
    wx, hx, c, n = size(x)
    outw, outh = pool_output_dims(size(x); window, pre_padding, post_padding, stride)
    xx = Float32.(Array(x))
    y = fill(-Inf32, outw, outh, c, n)
    for batch in 1:n, ch in 1:c, oy in 1:outh, ox in 1:outw
        for ky in 1:window[2], kx in 1:window[1]
            ix = (ox - 1) * stride[1] + kx - pre_padding[1]
            iy = (oy - 1) * stride[2] + ky - pre_padding[2]
            if 1 <= ix <= wx && 1 <= iy <= hx
                y[ox, oy, ch, batch] = max(y[ox, oy, ch, batch], xx[ix, iy, ch, batch])
            end
        end
    end
    return y
end

function meanpool2d_ref(x; window, pre_padding, post_padding, stride, include_pad::Bool)
    wx, hx, c, n = size(x)
    outw, outh = pool_output_dims(size(x); window, pre_padding, post_padding, stride)
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

function maxpool2d_bwd_ref(dy, x; window, pre_padding, post_padding, stride)
    wx, hx, c, n = size(x)
    ddy = Float32.(Array(dy))
    xx = Float32.(Array(x))
    dx = zeros(Float32, size(x))
    for batch in 1:n, ch in 1:c, oy in 1:size(dy, 2), ox in 1:size(dy, 1)
        best = -Inf32
        best_ix = 0
        best_iy = 0
        for ky in 1:window[2], kx in 1:window[1]
            ix = (ox - 1) * stride[1] + kx - pre_padding[1]
            iy = (oy - 1) * stride[2] + ky - pre_padding[2]
            if 1 <= ix <= wx && 1 <= iy <= hx && xx[ix, iy, ch, batch] > best
                best = xx[ix, iy, ch, batch]
                best_ix = ix
                best_iy = iy
            end
        end
        dx[best_ix, best_iy, ch, batch] += ddy[ox, oy, ch, batch]
    end
    return dx
end

function meanpool2d_bwd_ref(dy, x_size; window, pre_padding, post_padding, stride,
                            include_pad::Bool)
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

let W=7, H=6, C=2, N=2
    x = CuArray(reshape(Float32.(sin.(1:W*H*C*N)), W, H, C, N))
    window = (2, 3)
    pre_padding = (1, 0)
    post_padding = (0, 1)
    padding = (pre_padding[1], post_padding[1], pre_padding[2], post_padding[2])
    stride = (2, 1)
    alpha = 1.25f0
    beta = 0.5f0

    mean_ref = meanpool2d_ref(x; window, pre_padding, post_padding, stride,
                              include_pad=false)
    y0 = CuArray(reshape(Float32.(cos.(1:length(mean_ref))), size(mean_ref)))
    y = copy(y0)
    supported = true
    try
        meanpool!(y, x; window, padding, stride, alpha, beta, count_include_pad=false)
    catch e
        e isa UnsupportedGraphError || rethrow()
        @test_skip "meanpool! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Array(y) ≈ alpha .* mean_ref .+ beta .* Array(y0) rtol=1f-5 atol=1f-5
    end

    dy = CuArray(reshape(Float32.(cos.(1:length(mean_ref))), size(mean_ref)))
    dx0 = CuArray(reshape(Float32.(sin.(1:length(x))), size(x)) ./ 8)
    dx = copy(dx0)
    dx_ref = meanpool2d_bwd_ref(dy, size(x); window, pre_padding, post_padding, stride,
                                include_pad=false)
    supported = true
    try
        ∇meanpool!(dx, dy, y, x; window, padding, stride, alpha, beta,
                   count_include_pad=false)
    catch e
        e isa UnsupportedGraphError || rethrow()
        @test_skip "∇meanpool! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Array(dx) ≈ alpha .* dx_ref .+ beta .* Array(dx0) rtol=1f-5 atol=1f-5
    end

    max_ref = maxpool2d_ref(x; window, pre_padding, post_padding, stride)
    y0 = CuArray(reshape(Float32.(cos.(1:length(max_ref))), size(max_ref)))
    y = copy(y0)
    supported = true
    try
        maxpool!(y, x; window, padding, stride, alpha, beta)
    catch e
        e isa UnsupportedGraphError || rethrow()
        @test_skip "maxpool! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Array(y) ≈ alpha .* max_ref .+ beta .* Array(y0) rtol=1f-5 atol=1f-5
    end

    bwd_pre_padding = pre_padding
    bwd_post_padding = post_padding
    bwd_padding = padding
    max_ref = maxpool2d_ref(x; window, pre_padding=bwd_pre_padding,
                            post_padding=bwd_post_padding, stride)
    y = CUDA.zeros(Float32, size(max_ref))
    maxpool!(y, x; window, padding=bwd_padding, stride)
    dy = CuArray(reshape(Float32.(cos.(1:length(max_ref))), size(max_ref)))
    dx0 = CuArray(reshape(Float32.(sin.(1:length(x))), size(x)) ./ 8)
    dx = copy(dx0)
    dx_ref = maxpool2d_bwd_ref(dy, x; window, pre_padding=bwd_pre_padding,
                               post_padding=bwd_post_padding, stride)
    supported = true
    try
        ∇maxpool!(dx, dy, y, x; window, padding=bwd_padding, stride, alpha, beta)
    catch e
        e isa UnsupportedGraphError || rethrow()
        @test_skip "∇maxpool! graph engine is unsupported on this device"
        supported = false
    end
    if supported
        @test Array(dx) ≈ alpha .* dx_ref .+ beta .* Array(dx0) rtol=1f-5 atol=1f-5
    end
end
