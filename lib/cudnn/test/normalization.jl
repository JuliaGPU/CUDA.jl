using cuDNN: batchnorm_gradient!, batchnorm_gradient_supported,
             batchnorm_inference!, batchnorm_inference_supported,
             batchnorm_training!, batchnorm_training_supported
using BFloat16s: BFloat16

CUDACore.allowscalar(false)

let
    x = CUDACore.zeros(Float32, 4, 4, 2, 0)
    y = similar(x)
    param = CUDACore.zeros(Float32, 1, 1, 2, 1)
    @test batchnorm_inference_supported(y, x, param, param, param, param)
    @test batchnorm_inference!(y, x, param, param, param, param) === y
    # there are no batch statistics to compute
    @test !batchnorm_training_supported(y, x, param, param)
    @test_throws ArgumentError batchnorm_training!(y, x, param, param)
    @test !batchnorm_gradient_supported(similar(x), param, param, x, x, param, param, param)
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

function bn_var_ref(x)
    dims = Tuple([1:ndims(x)-2; ndims(x)])
    m = prod(size(x, d) for d in dims)
    mean = sum(x; dims) ./ m
    return sum(abs2, x .- mean; dims) ./ m, m
end

bn_rtol(::Type{Float64}) = 1e-10
bn_rtol(::Type{Float32}) = 1f-4
bn_rtol(::Type{Float16}) = 1f-2
bn_rtol(::Type{BFloat16}) = 5f-2

bn_types = capability(device()) >= v"8.0" ? (Float16, BFloat16, Float32, Float64) :
                                            (Float16, Float32, Float64)
@testset "batchnorm $T, size $sz" for T in bn_types,
                                       sz in ((3, 5), (4, 3, 5), (4, 3, 3, 5), (4, 3, 2, 3, 5))
    S = T === Float64 ? Float64 : Float32
    C = sz[end-1]
    pdims = ntuple(i -> i == length(sz) - 1 ? C : 1, length(sz))
    epsilon = 1e-4
    momentum = 0.25
    rtol = atol = bn_rtol(T)

    x_ref = S.(T.(reshape(sin.(1:prod(sz)), sz)))
    scale_ref = reshape(S.(range(0.5, 1.5; length=C)), pdims)
    bias_ref = reshape(S.(range(-0.25, 0.25; length=C)), pdims)
    y_ref, mean_ref, invvar_ref = bn_training_ref(x_ref, scale_ref, bias_ref; epsilon)

    x = CuArray(T.(x_ref))
    scale, bias = CuArray(vec(scale_ref)), CuArray(vec(bias_ref))
    @test batchnorm_training_supported(similar(x), x, scale, bias)
    @test !batchnorm_training_supported(similar(x), x, T === Float64 ? Float32.(scale) :
                                                         Float64.(scale), bias)

    # training, updating the running statistics
    rm0 = reshape(S.(range(-1, 1; length=C)), pdims)
    rv0 = reshape(S.(range(1, 2; length=C)), pdims)
    running_mean, running_var = CuArray(rm0), CuArray(rv0)
    y = similar(x)
    mean, invvar = batchnorm_training!(y, x, scale, bias; running_mean, running_var,
                                       momentum, epsilon)
    @test eltype(mean) == eltype(invvar) == S
    @test S.(Array(y)) ≈ y_ref rtol=rtol atol=atol
    @test vec(Array(mean)) ≈ vec(mean_ref) rtol=rtol atol=atol
    @test vec(Array(invvar)) ≈ vec(invvar_ref) rtol=rtol atol=atol
    var, m = bn_var_ref(x_ref)
    @test Array(running_mean) ≈ @.((1 - momentum) * rm0 + momentum * mean_ref) rtol=rtol atol=atol
    @test Array(running_var) ≈ @.((1 - momentum) * rv0 + momentum * var * m / (m - 1)) rtol=rtol atol=atol

    # blending into the output
    y0 = CuArray(T.(reshape(cos.(1:prod(sz)), sz)))
    y = copy(y0)
    batchnorm_training!(y, x, scale, bias; epsilon, alpha=2, beta=0.5)
    @test S.(Array(y)) ≈ 2 .* y_ref .+ 0.5 .* S.(Array(y0)) rtol=rtol atol=4atol

    # gradient
    dy_ref = S.(T.(reshape(cos.(1:prod(sz)), sz)))
    dx_ref, dscale_ref, dbias_ref = bn_backward_ref(dy_ref, x_ref, scale_ref, mean_ref,
                                                    invvar_ref)
    dx, dscale, dbias = similar(x), similar(scale), similar(bias)
    @test batchnorm_gradient_supported(dx, dscale, dbias, CuArray(T.(dy_ref)), x, scale,
                                       mean, invvar)
    batchnorm_gradient!(dx, dscale, dbias, CuArray(T.(dy_ref)), x, scale, mean, invvar;
                        epsilon)
    @test S.(Array(dx)) ≈ dx_ref rtol=rtol atol=atol
    @test vec(Array(dscale)) ≈ vec(dscale_ref) rtol=rtol atol=atol
    @test vec(Array(dbias)) ≈ vec(dbias_ref) rtol=rtol atol=atol

    # blending into the gradients
    dx0, dscale0, dbias0 = copy(dx), copy(dscale), copy(dbias)
    batchnorm_gradient!(dx, dscale, dbias, CuArray(T.(dy_ref)), x, scale, mean, invvar;
                        epsilon, alpha=2, beta=0.5, dalpha=3, dbeta=0.25)
    @test S.(Array(dx)) ≈ 2 .* dx_ref .+ 0.5 .* S.(Array(dx0)) rtol=rtol atol=4atol
    @test vec(Array(dscale)) ≈ 3 .* vec(dscale_ref) .+ 0.25 .* Array(dscale0) rtol=rtol atol=4atol
    @test vec(Array(dbias)) ≈ 3 .* vec(dbias_ref) .+ 0.25 .* Array(dbias0) rtol=rtol atol=4atol

    # inference with the running statistics
    yi = similar(x)
    @test batchnorm_inference_supported(yi, x, scale, bias, CuArray(rm0), CuArray(rv0))
    batchnorm_inference!(yi, x, scale, bias, CuArray(rm0), CuArray(rv0); epsilon)
    yi_ref = @. scale_ref * (x_ref - rm0) / sqrt(rv0 + epsilon) + bias_ref
    @test S.(Array(yi)) ≈ yi_ref rtol=rtol atol=atol
    yi0 = copy(yi)
    batchnorm_inference!(yi, x, scale, bias, CuArray(rm0), CuArray(rv0); epsilon, alpha=2,
                         beta=0.5)
    @test S.(Array(yi)) ≈ 2 .* yi_ref .+ 0.5 .* S.(Array(yi0)) rtol=rtol atol=4atol
end

if capability(device()) < v"8.0"
    let x = CuArray(rand(BFloat16, 4, 4, 3, 2)), p = CuArray(ones(Float32, 3))
        @test !batchnorm_training_supported(similar(x), x, p, p)
        @test_throws ArgumentError batchnorm_training!(similar(x), x, p, p)
    end
end

let x = CuArray(rand(Float32, 2, 2, 2, 2, 2, 3, 4)), p = CuArray(ones(Float32, 3))
    @test !batchnorm_training_supported(similar(x), x, p, p)
    @test_throws ArgumentError batchnorm_training!(similar(x), x, p, p)
end

let x = CuArray(rand(Float16, 4, 4, 3, 2)), p = CuArray(ones(Float16, 3))
    @test !batchnorm_training_supported(similar(x), x, p, p)
    @test_throws ArgumentError batchnorm_training!(similar(x), x, p, p)
end
