using cuDNN: maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, UnsupportedGraphError

CUDACore.allowscalar(false)

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

# with count_include_pad, the graph API's resampling engine used to count too little padding
# when the post padding exceeds the pre padding (#3473)
@testset "asymmetric padding, count_include_pad=$include_pad" for include_pad in (false, true)
    W, H, C, N = 7, 6, 2, 2
    x = CuArray(reshape(Float32.(sin.(1:W*H*C*N)), W, H, C, N))
    window = (2, 3)
    pre_padding = (1, 0)
    post_padding = (0, 1)
    padding = (pre_padding[1], post_padding[1], pre_padding[2], post_padding[2])
    stride = (2, 1)
    alpha = 1.25f0
    beta = 0.5f0

    mean_ref = meanpool2d_ref(x; window, pre_padding, post_padding, stride, include_pad)
    y0 = CuArray(reshape(Float32.(cos.(1:length(mean_ref))), size(mean_ref)))
    y = copy(y0)
    supported = true
    try
        meanpool!(y, x; window, padding, stride, alpha, beta, count_include_pad=include_pad)
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
                                include_pad)
    supported = true
    try
        ∇meanpool!(dx, dy, y, x; window, padding, stride, alpha, beta,
                   count_include_pad=include_pad)
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
    y = CUDACore.zeros(Float32, size(max_ref))
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

import NNlib

# the NNlib in the test environment predates the fix for its CPU `∇maxpool` picking the
# wrong element of a window when values nearly tie (FluxML/NNlib.jl#775)
function maxpool_bwd_ref(dy, x; window, padding, stride)
    nd = ndims(x) - 2
    dx = zero(x)
    for b in axes(x, nd+2), c in axes(x, nd+1), o in CartesianIndices(size(dy)[1:nd])
        best, besti = -Inf32, nothing
        for k in CartesianIndices(ntuple(_ -> window, nd))
            i = ntuple(d -> (o[d] - 1) * stride + k[d] - padding, nd)
            all(d -> 1 <= i[d] <= size(x, d), 1:nd) || continue
            x[i..., c, b] > best && ((best, besti) = (x[i..., c, b], i))
        end
        dx[besti..., c, b] += dy[o, c, b]
    end
    return dx
end

# symmetric padding uses the fixed-function API, which also supports 1D pooling
@testset "symmetric $(nd)D pooling $T" for nd in (1, 2, 3), T in (Float32, Float16)
    rtol = atol = T === Float16 ? 1f-2 : 1f-5
    xsize = (ntuple(_ -> 7, nd)..., 3, 2)
    x_h = T.(reshape(sin.(1:prod(xsize)), xsize))
    x = CuArray(x_h)
    for (window, padding, stride) in ((2, 0, 2), (3, 1, 2))
        pdims = NNlib.PoolDims(x_h, window; padding, stride)

        y_ref = NNlib.maxpool(Float32.(x_h), pdims)
        y = CuArray{T}(undef, size(y_ref))
        maxpool!(y, x; window, padding, stride)
        @test Float32.(Array(y)) ≈ y_ref rtol=rtol atol=atol

        dy_h = T.(reshape(cos.(1:length(y_ref)), size(y_ref)))
        dx_ref = maxpool_bwd_ref(Float32.(dy_h), Float32.(x_h); window, padding, stride)
        for deterministic in (false, true)
            dx = similar(x)
            ∇maxpool!(dx, CuArray(dy_h), y, x; window, padding, stride, deterministic)
            @test Float32.(Array(dx)) ≈ dx_ref rtol=rtol atol=atol
        end

        y_ref = NNlib.meanpool(Float32.(x_h), pdims)
        meanpool!(y, x; window, padding, stride)
        @test Float32.(Array(y)) ≈ y_ref rtol=rtol atol=atol
        y0 = copy(y)
        meanpool!(y, x; window, padding, stride, alpha=2, beta=0.5)
        @test Float32.(Array(y)) ≈ 2 .* y_ref .+ 0.5 .* Float32.(Array(y0)) rtol=rtol atol=4atol

        dx_ref = NNlib.∇meanpool(Float32.(dy_h), y_ref, Float32.(x_h), pdims)
        dx = similar(x)
        ∇meanpool!(dx, CuArray(dy_h), y, x; window, padding, stride)
        @test Float32.(Array(dx)) ≈ dx_ref rtol=rtol atol=atol
        dx0 = copy(dx)
        ∇meanpool!(dx, CuArray(dy_h), y, x; window, padding, stride, alpha=2, beta=0.5)
        @test Float32.(Array(dx)) ≈ 2 .* dx_ref .+ 0.5 .* Float32.(Array(dx0)) rtol=rtol atol=4atol
    end

    @test_throws DimensionMismatch maxpool!(similar(x), x; window=2)
end

let x = CuArray(reshape(Float32.(sin.(1:7*6*2*2)), 7, 6, 2, 2))
    window = (2, 3)
    pre_padding = post_padding = (1, 1)
    stride = (2, 1)
    y_ref = meanpool2d_ref(x; window, pre_padding, post_padding, stride, include_pad=false)
    y = CuArray{Float32}(undef, size(y_ref))
    meanpool!(y, x; window, padding=1, stride, count_include_pad=false)
    @test Array(y) ≈ y_ref rtol=1f-5 atol=1f-5
end

# NaN propagation must not depend on whether the padding is symmetric
@testset "NaN propagation, padding $padding" for padding in (0, (0, 1, 0, 1))
    x_h = reshape(Float32.(1:4*4*2*2), 4, 4, 2, 2)
    x_h[1, 1, 1, 1] = NaN
    x = CuArray(x_h)
    pre, post = padding == 0 ? ((0, 0), (0, 0)) : ((0, 0), (1, 1))
    ysize = pool_output_dims(size(x); window=(2, 2), pre_padding=pre, post_padding=post,
                             stride=(2, 2))
    y = CuArray{Float32}(undef, ysize..., 2, 2)
    maxpool!(y, x; window=2, padding)
    @test isnan(Array(y)[1, 1, 1, 1])
    try
        maxpool!(y, x; window=2, padding, nan_propagation=cuDNN.CUDNN_NOT_PROPAGATE_NAN)
        @test Array(y)[1, 1, 1, 1] == maximum(filter(!isnan, x_h[1:2, 1:2, 1, 1]))
    catch e
        e isa UnsupportedGraphError || rethrow()
        @test_skip "maxpool! graph engine without NaN propagation is unsupported"
    end
end

# a range of channels is pooled in place, by the fixed-function API and by graphs
@testset "views, padding $padding" for padding in (1, (1, 0, 0, 1))
    C = 3
    channels(a) = view(a, :, :, C+1:2C, :)
    x = channels(CuArray(rand(Float32, 7, 7, 3C, 2)))
    xd = copy(x)
    yd = similar(xd, 4, 4, C, 2)
    y = channels(similar(xd, 4, 4, 3C, 2))
    dy = channels(CuArray(rand(Float32, 4, 4, 3C, 2)))
    dx = channels(similar(xd, 7, 7, 3C, 2))
    for (pool!, ∇pool!) in ((maxpool!, ∇maxpool!), (meanpool!, ∇meanpool!))
        pool!(yd, xd; window=2, padding)
        pool!(y, x; window=2, padding)
        @test Array(y) ≈ Array(yd)
        dxd = ∇pool!(similar(xd), copy(dy), yd, xd; window=2, padding)
        ∇pool!(dx, dy, y, x; window=2, padding)
        @test Array(dx) ≈ Array(dxd)
    end

    # reversed views, which cuDNN can't address, are copied
    reversed(a) = view(a, :, :, size(a, 3):-1:1, :)
    xr, dyr = reversed(copy(x)), reversed(copy(dy))
    y0, dx0 = CuArray(rand(Float32, size(yd))), CuArray(rand(Float32, size(xd)))
    for (pool!, ∇pool!) in ((maxpool!, ∇maxpool!), (meanpool!, ∇meanpool!))
        # NaN-filled, so that reading the old contents with `beta == 0` would show
        yr = reversed(fill!(similar(yd), NaN))
        pool!(yr, xr; window=2, padding)
        @test Array(yr) ≈ Array(pool!(similar(yd), copy(xr); window=2, padding))
        dxr = reversed(fill!(similar(xd), NaN))
        ∇pool!(dxr, dyr, yr, xr; window=2, padding)
        @test Array(dxr) ≈ Array(∇pool!(similar(xd), copy(dyr), copy(yr), copy(xr);
                                        window=2, padding))

        # a staged output keeps the contents that `beta` scales
        yb = copyto!(reversed(similar(yd)), y0)
        pool!(yb, xr; window=2, padding, beta=2)
        @test Array(yb) ≈ Array(pool!(copy(y0), copy(xr); window=2, padding, beta=2))
        dxb = copyto!(reversed(similar(xd)), dx0)
        ∇pool!(dxb, dyr, yr, xr; window=2, padding, beta=2)
        @test Array(dxb) ≈ Array(∇pool!(copy(dx0), copy(dyr), copy(yr), copy(xr);
                                        window=2, padding, beta=2))
    end
end

# max-pooling gradients laid out unlike the forward arrays are copied, as cuDNN rejects them,
# or with graphs, computes wrong gradients (#3497)
@testset "∇maxpool! layouts, padding $padding" for padding in (1, (1, 0, 0, 1))
    C = 3
    # a range of channels, with strides that depend on the size of the parent
    channels(a, n) = copyto!(view(similar(a, size(a)[1:2]..., n*size(a, 3), size(a, 4)),
                                  :, :, size(a, 3)+1:2size(a, 3), :), a)
    xd = CuArray(rand(Float32, 7, 7, C, 2))
    yd = maxpool!(similar(xd, 4, 4, C, 2), xd; window=2, padding)
    dyd = CuArray(rand(Float32, size(yd)))
    dx0 = CuArray(rand(Float32, size(xd)))
    ref = Array(∇maxpool!(similar(xd), dyd, yd, xd; window=2, padding))
    for (x, dx, y, dy) in ((channels(xd, 3), copy(dx0), yd, dyd),
                           (xd, channels(dx0, 3), yd, dyd),
                           (xd, copy(dx0), channels(yd, 3), dyd),
                           (xd, copy(dx0), yd, channels(dyd, 3)),
                           (channels(xd, 3), channels(dx0, 5),
                            channels(yd, 3), channels(dyd, 5)))
        @test Array(∇maxpool!(dx, dy, y, x; window=2, padding)) ≈ ref
    end
    dx = channels(dx0, 3)
    ∇maxpool!(dx, dyd, yd, xd; window=2, padding, beta=2)
    @test Array(dx) ≈ ref .+ 2 .* Array(dx0)
end
