using CUDA
using cuDNN: batchnorm_gradient!, batchnorm_gradient_supported,
             batchnorm_inference!, batchnorm_inference_supported,
             batchnorm_training!, batchnorm_training_supported, graph_unsupported

CUDA.allowscalar(false)

let
    x = CUDA.zeros(Float32, 4, 4, 2, 0)
    y = similar(x)
    param = CUDA.zeros(Float32, 1, 1, 2, 1)
    @test batchnorm_inference_supported(y, x, param, param, param, param)
    @test batchnorm_inference!(y, x, param, param, param, param) === y
end

function bn_reduce_dims(x)
    ntuple(i -> i == ndims(x) - 1 ? Colon() : axes(x, i), ndims(x))
end

function bn_stats_ref(x; epsilon)
    dims = Tuple([1:ndims(x)-2; ndims(x)])
    mean = sum(x; dims) ./ prod(size(x, d) for d in dims)
    var = sum(abs2, x .- mean; dims) ./ prod(size(x, d) for d in dims)
    invvar = @. 1 / sqrt(var + epsilon)
    return mean, invvar
end

function bn_training_ref(x, scale, bias; epsilon)
    mean, invvar = bn_stats_ref(x; epsilon)
    y = @. scale * (x - mean) * invvar + bias
    return y, mean, invvar
end

function bn_backward_ref(dy, x, scale, mean, invvar)
    rdims = Tuple([1:ndims(x)-2; ndims(x)])
    m = prod(size(x, d) for d in rdims)
    xhat = @. (x - mean) * invvar
    dbias = sum(dy; dims=rdims)
    dscale = sum(dy .* xhat; dims=rdims)
    dx = @. scale * invvar / m * (m * dy - dbias - xhat * dscale)
    return dx, dscale, dbias
end

let W=4, H=3, C=2, N=3
    epsilon = 1f-4
    x_ref = reshape(Float32.(sin.(1:W*H*C*N)), W, H, C, N)
    scale_ref = reshape(Float32[1.5, 0.75], 1, 1, C, 1)
    bias_ref = reshape(Float32[-0.25, 0.5], 1, 1, C, 1)
    y_ref, mean_ref, invvar_ref = bn_training_ref(x_ref, scale_ref, bias_ref; epsilon)

    x = CuArray(Float16.(x_ref))
    scale, bias = CuArray(scale_ref), CuArray(bias_ref)
    y = similar(x)
    @test_throws ArgumentError batchnorm_training!(y, x, Float16.(scale), Float16.(bias);
                                                    epsilon)
    try
        mean, invvar = batchnorm_training!(y, x, scale, bias; epsilon)
        @test eltype(mean) == Float32
        @test eltype(invvar) == Float32
        @test Float32.(Array(y)) ≈ y_ref rtol=3f-3 atol=3f-3
        @test Array(mean) ≈ mean_ref rtol=3f-3 atol=3f-3
        @test Array(invvar) ≈ invvar_ref rtol=3f-3 atol=3f-3

        dy = CuArray(Float16.(reshape(cos.(1:length(x_ref)), size(x_ref))))
        dx, dscale, dbias = similar(x), similar(scale), similar(bias)
        batchnorm_gradient!(dx, dscale, dbias, dy, x, scale, mean, invvar; epsilon)
        @test eltype(dx) == Float16
        @test eltype(dscale) == Float32
        @test eltype(dbias) == Float32
    catch e
        graph_unsupported(e) || rethrow()
        @test_skip "Float16 batchnorm graph engine is unsupported on this device"
    end
end

let W=4, H=3, C=2, N=3
    epsilon = 1f-4
    x_h = reshape(Float32.(sin.(1:W*H*C*N)), W, H, C, N)
    scale_h = reshape(Float32[1.5, 0.75], 1, 1, C, 1)
    bias_h = reshape(Float32[-0.25, 0.5], 1, 1, C, 1)
    y_ref, mean_ref, invvar_ref = bn_training_ref(x_h, scale_h, bias_h; epsilon)

    x, scale, bias = CuArray(x_h), CuArray(scale_h), CuArray(bias_h)
    y = similar(x)
    running_mean = CuArray(mean_ref)
    running_var = CuArray(@. 1 / invvar_ref^2 - epsilon)
    @test_throws ArgumentError batchnorm_training!(y, x, scale, bias; alpha=2)
    @test_throws ArgumentError batchnorm_inference!(y, x, scale, bias, running_mean,
                                                     running_var; beta=1)
    @test_throws ArgumentError batchnorm_gradient!(similar(x), similar(scale),
                                                    similar(bias), x, x, scale,
                                                    running_mean, running_var; dalpha=2)
    @test batchnorm_training_supported(y, x, scale, bias) isa Bool
    @test batchnorm_inference_supported(y, x, scale, bias, running_mean,
                                        running_var) isa Bool
    @test batchnorm_gradient_supported(similar(x), similar(scale), similar(bias), x,
                                       x, scale, running_mean, running_var) isa Bool
    try
        mean, invvar = batchnorm_training!(y, x, scale, bias; epsilon)

        @test Array(y) ≈ y_ref rtol=1f-5 atol=1f-5
        @test Array(mean) ≈ mean_ref rtol=1f-5 atol=1f-5
        @test Array(invvar) ≈ invvar_ref rtol=1f-5 atol=1f-5

        dy_h = reshape(Float32.(cos.(1:length(x_h))), size(x_h))
        dx_ref, dscale_ref, dbias_ref = bn_backward_ref(dy_h, x_h, scale_h, mean_ref,
                                                        invvar_ref)
        dx, dscale, dbias = similar(x), similar(scale), similar(bias)
        batchnorm_gradient!(dx, dscale, dbias, CuArray(dy_h), x, scale, mean, invvar;
                            epsilon)
        @test Array(dx) ≈ dx_ref rtol=1f-4 atol=1f-4
        @test Array(dscale) ≈ dscale_ref rtol=1f-4 atol=1f-4
        @test Array(dbias) ≈ dbias_ref rtol=1f-4 atol=1f-4

        yi = similar(x)
        batchnorm_inference!(yi, x, scale, bias, running_mean, running_var; epsilon)
        @test Array(yi) ≈ y_ref rtol=1f-5 atol=1f-5
    catch e
        graph_unsupported(e) || rethrow()
        @test_skip "batchnorm graph engine is unsupported on this device"
    end
end
