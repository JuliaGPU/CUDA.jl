bn_param_size(x::AbstractArray{<:Any,N}) where {N} =
    ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

function check_bn_running_stats(running_mean, running_var)
    (running_mean === nothing) == (running_var === nothing) ||
        throw(ArgumentError("running_mean and running_var must both be arrays or both be nothing"))
    return
end

function check_bn_eltype(name, a::DenseCuArray, T)
    eltype(a) == T || throw(ArgumentError("$name must have $T eltype"))
    return a
end

bn_stat_type(::Type{Float64}) = Float64
bn_stat_type(::Type) = Float32

# cuDNN only ships BFloat16 batch normalization kernels for compute capability 8.0+, and
# fails with CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH on older devices.
bn_supported_eltype(::Type{T}) where {T} =
    T in (Float16, Float32, Float64) ||
    (T === BFloat16 && capability(device()) >= v"8.0")

function check_bn_input(y::DenseCuArray{T}, x::DenseCuArray{T}) where {T}
    bn_supported_eltype(T) ||
        throw(ArgumentError("batchnorm does not support $T inputs on this device"))
    2 <= ndims(x) <= 5 ||
        throw(ArgumentError("batchnorm input must have rank 2 to 5, got $(ndims(x))"))
    size(y) == size(x) ||
        throw(DimensionMismatch("batchnorm output has size $(size(y)), expected $(size(x))"))
    return
end

function bn_param_array(a::DenseCuArray, x::DenseCuArray)
    dims = bn_param_size(x)
    size(a) == dims && return a
    length(a) == prod(dims) ||
        throw(DimensionMismatch("batchnorm parameter must have size $dims or length $(prod(dims))"))
    return reshape(a, dims)
end

# `dim4` gives rank 2 and 3 inputs singleton spatial dimensions, as cuDNN only normalizes
# tensors of rank 4 and 5.
function bn_param_desc(x::DenseCuArray{T}) where {T}
    dims = dim4(bn_param_size(x), Val(CUDNN_TENSOR_NCHW))
    cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(bn_stat_type(T)),
                          Cint(length(dims)), dims)
end

# The graph API has no batch-normalization engine for inference, Float64, or channel
# counts that are not a multiple of 4 (8 for 16-bit types), and where it has one it is
# about twice as slow as the fixed-function API (cuDNN 9.26, #3380).
function batchnorm_training!(y::DenseCuArray{T}, x::DenseCuArray{T},
                             scale::DenseCuArray, bias::DenseCuArray;
                             running_mean=nothing, running_var=nothing,
                             momentum::Real=0.1, epsilon::Real=1e-5,
                             alpha::Real=1, beta::Real=0,
                             deterministic::Bool=false,
                             math_mode=CUDACore.math_mode(),
                             max_workspace::Union{Nothing,Integer}=nothing) where {T}
    check_bn_input(y, x)
    isempty(x) && throw(ArgumentError("batchnorm_training! requires a non-empty input"))
    check_bn_running_stats(running_mean, running_var)
    S = bn_stat_type(T)
    check_bn_eltype("scale", scale, S)
    check_bn_eltype("bias", bias, S)
    running_mean === nothing || check_bn_eltype("running_mean", running_mean, S)
    running_var === nothing || check_bn_eltype("running_var", running_var, S)
    saved_mean = similar(scale, S, size(scale))
    saved_invvar = similar(scale, S, size(scale))
    running(a) = a === nothing ? CU_NULL : bn_param_array(a, x)
    cudnnBatchNormalizationForwardTraining(
        handle(), CUDNN_BATCHNORM_SPATIAL, scalingParameter(T, alpha),
        scalingParameter(T, beta), cudnnTensorDescriptor(x), x, cudnnTensorDescriptor(y),
        y, bn_param_desc(x), bn_param_array(scale, x), bn_param_array(bias, x),
        Float64(momentum), running(running_mean), running(running_var),
        Float64(max(epsilon, CUDNN_BN_MIN_EPSILON)), bn_param_array(saved_mean, x),
        bn_param_array(saved_invvar, x))
    return saved_mean, saved_invvar
end

function batchnorm_inference!(y::DenseCuArray{T}, x::DenseCuArray{T},
                              scale::DenseCuArray, bias::DenseCuArray,
                              running_mean::DenseCuArray, running_var::DenseCuArray;
                              epsilon::Real=1e-5, alpha::Real=1, beta::Real=0,
                              deterministic::Bool=false,
                              math_mode=CUDACore.math_mode(),
                              max_workspace::Union{Nothing,Integer}=nothing) where {T}
    check_bn_input(y, x)
    isempty(y) && return y
    S = bn_stat_type(T)
    for (name, a) in (("scale", scale), ("bias", bias), ("running_mean", running_mean),
                      ("running_var", running_var))
        check_bn_eltype(name, a, S)
    end
    cudnnBatchNormalizationForwardInference(
        handle(), CUDNN_BATCHNORM_SPATIAL, scalingParameter(T, alpha),
        scalingParameter(T, beta), cudnnTensorDescriptor(x), x, cudnnTensorDescriptor(y),
        y, bn_param_desc(x), bn_param_array(scale, x), bn_param_array(bias, x),
        bn_param_array(running_mean, x), bn_param_array(running_var, x),
        Float64(max(epsilon, CUDNN_BN_MIN_EPSILON)))
    return y
end

function batchnorm_gradient!(dx::DenseCuArray{T}, dscale::DenseCuArray,
                             dbias::DenseCuArray, dy::DenseCuArray{T},
                             x::DenseCuArray{T}, scale::DenseCuArray,
                             saved_mean::DenseCuArray, saved_invvar::DenseCuArray;
                             epsilon::Real=1e-5, alpha::Real=1, beta::Real=0,
                             dalpha::Real=1, dbeta::Real=0,
                             deterministic::Bool=false,
                             math_mode=CUDACore.math_mode(),
                             max_workspace::Union{Nothing,Integer}=nothing) where {T}
    check_bn_input(dx, x)
    isempty(x) && throw(ArgumentError("batchnorm_gradient! requires a non-empty input"))
    size(dy) == size(x) ||
        throw(DimensionMismatch("batchnorm dy has size $(size(dy)), expected $(size(x))"))
    S = bn_stat_type(T)
    for (name, a) in (("scale", scale), ("dscale", dscale), ("dbias", dbias),
                      ("saved_mean", saved_mean), ("saved_invvar", saved_invvar))
        check_bn_eltype(name, a, S)
    end
    cudnnBatchNormalizationBackward(
        handle(), CUDNN_BATCHNORM_SPATIAL, scalingParameter(T, alpha),
        scalingParameter(T, beta), scalingParameter(T, dalpha), scalingParameter(T, dbeta),
        cudnnTensorDescriptor(x), x, cudnnTensorDescriptor(dy), dy,
        cudnnTensorDescriptor(dx), dx, bn_param_desc(x), bn_param_array(scale, x),
        bn_param_array(dscale, x), bn_param_array(dbias, x),
        Float64(max(epsilon, CUDNN_BN_MIN_EPSILON)), bn_param_array(saved_mean, x),
        bn_param_array(saved_invvar, x))
    return dx, dscale, dbias
end

bn_supported(x::DenseCuArray{T}, params...) where {T} =
    bn_supported_eltype(T) && 2 <= ndims(x) <= 5 &&
    all(a -> eltype(a) == bn_stat_type(T), params)

function batchnorm_training_supported(y::DenseCuArray{T}, x::DenseCuArray{T},
                                      scale::DenseCuArray, bias::DenseCuArray;
                                      running_mean=nothing, running_var=nothing,
                                      momentum::Real=0.1, epsilon::Real=1e-5,
                                      deterministic::Bool=false,
                                      math_mode=CUDACore.math_mode(),
                                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    check_bn_running_stats(running_mean, running_var)
    stats = running_mean === nothing ? () : (running_mean, running_var)
    return !isempty(x) && bn_supported(x, scale, bias, stats...)
end

function batchnorm_inference_supported(y::DenseCuArray{T}, x::DenseCuArray{T},
                                       scale::DenseCuArray, bias::DenseCuArray,
                                       running_mean::DenseCuArray,
                                       running_var::DenseCuArray;
                                       deterministic::Bool=false,
                                       math_mode=CUDACore.math_mode(),
                                       max_workspace::Union{Nothing,Integer}=nothing) where {T}
    return bn_supported(x, scale, bias, running_mean, running_var)
end

function batchnorm_gradient_supported(dx::DenseCuArray{T}, dscale::DenseCuArray,
                                      dbias::DenseCuArray, dy::DenseCuArray{T},
                                      x::DenseCuArray{T}, scale::DenseCuArray,
                                      saved_mean::DenseCuArray,
                                      saved_invvar::DenseCuArray;
                                      deterministic::Bool=false,
                                      math_mode=CUDACore.math_mode(),
                                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    return !isempty(x) && bn_supported(x, scale, dscale, dbias, saved_mean, saved_invvar)
end

@doc raw"""
    batchnorm_training!(y, x, scale, bias; running_mean=nothing, running_var=nothing,
                        momentum=0.1, epsilon=1e-5, alpha=1, beta=0)
        -> (saved_mean, saved_invvar)
    batchnorm_inference!(y, x, scale, bias, running_mean, running_var; epsilon=1e-5,
                         alpha=1, beta=0)
    batchnorm_gradient!(dx, dscale, dbias, dy, x, scale, saved_mean, saved_invvar;
                        epsilon=1e-5, alpha=1, beta=0, dalpha=1, dbeta=0)
    batchnorm_training_supported(y, x, scale, bias; kwargs...) -> Bool
    batchnorm_inference_supported(y, x, scale, bias, running_mean, running_var;
                                  kwargs...) -> Bool
    batchnorm_gradient_supported(dx, dscale, dbias, dy, x, scale, saved_mean,
                                 saved_invvar; kwargs...) -> Bool

Spatial batch normalization, normalizing over all dimensions of `x` except the channel
dimension (the second-to-last). `x` must have rank 2 to 5. Parameters may be vectors of
length `C` or shaped `(1, ..., C, 1)`.

`Float16` and `BFloat16` inputs use `Float32` parameters and statistics. `Float32` and
`Float64` inputs use parameters and statistics of the same type. `BFloat16` requires a
device with compute capability 8.0 or newer.

Training returns the saved batch statistics that the gradient consumes, with
`saved_invvar = 1 / sqrt(var + epsilon)`, and updates `running_mean`/`running_var` in
place when given, using the unbiased batch variance: `running = (1 - momentum) * running +
momentum * batch`. `alpha` and `beta` blend the result into the output (`y = alpha * bn(x)
+ beta * y`); for the gradient, `dalpha` and `dbeta` do the same for `dscale` and
`dbias`.

These operations use cuDNN's fixed-function batch normalization, as the graph API only
supports it in fused, channels-last training patterns. The `deterministic`, `math_mode`,
and `max_workspace` keywords are accepted for consistency with the other operations; the
computation is always deterministic.
"""
batchnorm_training!, batchnorm_inference!, batchnorm_gradient!,
batchnorm_training_supported, batchnorm_inference_supported,
batchnorm_gradient_supported

@public batchnorm_training!, batchnorm_inference!, batchnorm_gradient!,
        batchnorm_training_supported, batchnorm_inference_supported,
        batchnorm_gradient_supported
