using CUDA
using cuDNN: batchnorm_gradient!, batchnorm_inference!, batchnorm_training!

CUDA.allowscalar(false)

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
    x_h = reshape(Float32.(sin.(1:W*H*C*N)), W, H, C, N)
    scale_h = reshape(Float32[1.5, 0.75], 1, 1, C, 1)
    bias_h = reshape(Float32[-0.25, 0.5], 1, 1, C, 1)
    y_ref, mean_ref, invvar_ref = bn_training_ref(x_h, scale_h, bias_h; epsilon)

    x, scale, bias = CuArray(x_h), CuArray(scale_h), CuArray(bias_h)
    y = similar(x)
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

    running_mean = CuArray(mean_ref)
    running_var = CuArray(@. 1 / invvar_ref^2 - epsilon)
    yi = similar(x)
    batchnorm_inference!(yi, x, scale, bias, running_mean, running_var; epsilon)
    @test Array(yi) ≈ y_ref rtol=1f-5 atol=1f-5
end
