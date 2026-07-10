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

bn_compute_type(::Type{Float64}) = Float64
bn_compute_type(::Type) = Float32

bn_stat_type(::Type{Float64}) = Float64
bn_stat_type(::Type) = Float32

function bn_param_array(a::DenseCuArray, x::DenseCuArray)
    dims = bn_param_size(x)
    size(a) == dims && return a
    length(a) == prod(dims) ||
        throw(DimensionMismatch("batchnorm parameter must have size $dims or length $(prod(dims))"))
    return reshape(a, dims)
end

bn_array_key(a::DenseCuArray) = (size(a), strides(a), pointer_alignment(a))

function bn_training_key(y, x, scale, bias, saved_mean, saved_invvar,
                          running_mean, running_var, epsilon, momentum,
                          deterministic, math_mode, max_workspace)
    (:bn_training, eltype(x), bn_array_key(y), bn_array_key(x), bn_array_key(scale),
     bn_array_key(bias), bn_array_key(saved_mean), bn_array_key(saved_invvar),
     running_mean === nothing ? nothing : bn_array_key(running_mean),
     running_var === nothing ? nothing : bn_array_key(running_var),
     Float64(epsilon), Float64(momentum), deterministic, math_mode, max_workspace)
end

function bn_inference_key(y, x, scale, bias, mean, stat_template, deterministic,
                           math_mode, max_workspace)
    (:bn_inference, eltype(x), bn_array_key(y), bn_array_key(x), bn_array_key(scale),
     bn_array_key(bias), bn_array_key(mean), bn_array_key(stat_template), deterministic,
     math_mode, max_workspace)
end

function bn_gradient_key(dx, dscale, dbias, dy, x, scale, mean, invvar, deterministic,
                          math_mode, max_workspace)
    (:bn_gradient, eltype(x), bn_array_key(dx), bn_array_key(dscale), bn_array_key(dbias),
     bn_array_key(dy), bn_array_key(x), bn_array_key(scale), bn_array_key(mean),
     bn_array_key(invvar), deterministic, math_mode, max_workspace)
end

function build_batchnorm_training_graph(y, x, scale, bias, saved_mean, saved_invvar,
                                         running_mean, running_var;
                                         deterministic, math_mode, max_workspace)
    ctype = bn_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tx = tensor!(g, x; name="X")
    tscale = tensor!(g, scale; name="Scale")
    tbias = tensor!(g, bias; name="Bias")
    ty = tensor!(g, y; name="Y", output=true)
    tmean = tensor!(g, saved_mean; name="Mean", output=true)
    tinv = tensor!(g, saved_invvar; name="InvVariance", output=true)
    teps = scalar!(g, ctype; rank=ndims(x), name="Epsilon")
    if running_mean === nothing
        norm_fwd!(g, tx, tscale, tbias; y=ty, mean=tmean, inv_variance=tinv,
                  epsilon=teps, phase=:training)
    else
        trmean_in = tensor!(g, running_mean; name="RunningMeanIn")
        trvar_in = tensor!(g, running_var; name="RunningVarIn")
        trmean_out = tensor!(g, running_mean; name="RunningMeanOut", output=true)
        trvar_out = tensor!(g, running_var; name="RunningVarOut", output=true)
        tmomentum = scalar!(g, ctype; rank=ndims(x), name="Momentum")
        norm_fwd!(g, tx, tscale, tbias; y=ty, mean=tmean, inv_variance=tinv,
                  epsilon=teps, momentum=tmomentum, input_running_mean=trmean_in,
                  input_running_var=trvar_in, output_running_mean=trmean_out,
                  output_running_var=trvar_out, phase=:training)
    end
    build!(g; deterministic, math_mode, max_workspace)
end

function build_batchnorm_inference_graph(y, x, scale, bias, mean, stat_template;
                                          deterministic, math_mode, max_workspace)
    ctype = bn_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tx = tensor!(g, x; name="X")
    tscale = tensor!(g, scale; name="Scale")
    tbias = tensor!(g, bias; name="Bias")
    tmean = tensor!(g, mean; name="Mean")
    tinv = tensor!(g, stat_template; name="InvVariance")
    ty = tensor!(g, y; name="Y", output=true)
    norm_fwd!(g, tx, tscale, tbias; y=ty, mean=tmean, inv_variance=tinv,
              phase=:inference)
    build!(g; deterministic, math_mode, max_workspace)
end

function build_batchnorm_gradient_graph(dx, dscale, dbias, dy, x, scale, mean, invvar;
                                         deterministic, math_mode, max_workspace)
    ctype = bn_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tdy = tensor!(g, dy; name="dY")
    tx = tensor!(g, x; name="X")
    tscale = tensor!(g, scale; name="Scale")
    tmean = tensor!(g, mean; name="Mean")
    tinv = tensor!(g, invvar; name="InvVariance")
    tdx = tensor!(g, dx; name="dX", output=true)
    tdscale = tensor!(g, dscale; name="dScale", output=true)
    tdbias = tensor!(g, dbias; name="dBias", output=true)
    norm_bwd!(g, tdy, tx, tscale, tmean, tinv; dx=tdx, dscale=tdscale,
              dbias=tdbias)
    build!(g; deterministic, math_mode, max_workspace)
end

function batchnorm_training!(y::DenseCuArray{T}, x::DenseCuArray{T},
                             scale::DenseCuArray, bias::DenseCuArray;
                             running_mean=nothing, running_var=nothing,
                             momentum::Real=0.1, epsilon::Real=1e-5,
                             alpha::Real=1, beta::Real=0,
                             deterministic::Bool=false,
                             math_mode=CUDACore.math_mode(),
                             max_workspace::Union{Nothing,Integer}=nothing) where {T}
    alpha == 1 && beta == 0 ||
        throw(ArgumentError("batchnorm_training! requires alpha=1 and beta=0"))
    epsilon = max(epsilon, CUDNN_BN_MIN_EPSILON)
    check_bn_running_stats(running_mean, running_var)
    S = bn_stat_type(T)
    check_bn_eltype("scale", scale, S)
    check_bn_eltype("bias", bias, S)
    running_mean === nothing || check_bn_eltype("running_mean", running_mean, S)
    running_var === nothing || check_bn_eltype("running_var", running_var, S)
    ps = bn_param_size(x)
    saved_mean = similar(scale, S, size(scale))
    saved_invvar = similar(scale, S, size(scale))
    scale_p, bias_p = bn_param_array(scale, x), bn_param_array(bias, x)
    saved_mean_p, saved_invvar_p = reshape(saved_mean, ps), reshape(saved_invvar, ps)
    rm_p = running_mean === nothing ? nothing : bn_param_array(running_mean, x)
    rv_p = running_var === nothing ? nothing : bn_param_array(running_var, x)
    key = bn_training_key(y, x, scale_p, bias_p, saved_mean_p, saved_invvar_p, rm_p,
                           rv_p, epsilon, momentum, deterministic, math_mode,
                           max_workspace)
    g = cached_graph(key) do
        build_batchnorm_training_graph(y, x, scale_p, bias_p, saved_mean_p,
                                        saved_invvar_p, rm_p, rv_p; deterministic,
                                        math_mode, max_workspace)
    end
    bindings = IdDict{Tensor,Any}(
        tensor(g, "X") => x,
        tensor(g, "Scale") => scale_p,
        tensor(g, "Bias") => bias_p,
        tensor(g, "Y") => y,
        tensor(g, "Mean") => saved_mean_p,
        tensor(g, "InvVariance") => saved_invvar_p,
        tensor(g, "Epsilon") => bn_compute_type(T)(epsilon),
    )
    if rm_p !== nothing
        bindings[tensor(g, "RunningMeanIn")] = rm_p
        bindings[tensor(g, "RunningVarIn")] = rv_p
        bindings[tensor(g, "RunningMeanOut")] = rm_p
        bindings[tensor(g, "RunningVarOut")] = rv_p
        bindings[tensor(g, "Momentum")] = bn_compute_type(T)(momentum)
    end
    execute!(g, bindings)
    return saved_mean, saved_invvar
end

function batchnorm_inference!(y::DenseCuArray{T}, x::DenseCuArray{T},
                              scale::DenseCuArray, bias::DenseCuArray,
                              running_mean::DenseCuArray, running_var::DenseCuArray;
                              epsilon::Real=1e-5, alpha::Real=1, beta::Real=0,
                              deterministic::Bool=false,
                              math_mode=CUDACore.math_mode(),
                              max_workspace::Union{Nothing,Integer}=nothing) where {T}
    alpha == 1 && beta == 0 ||
        throw(ArgumentError("batchnorm_inference! requires alpha=1 and beta=0"))
    epsilon = max(epsilon, CUDNN_BN_MIN_EPSILON)
    S = bn_stat_type(T)
    check_bn_eltype("scale", scale, S)
    check_bn_eltype("bias", bias, S)
    check_bn_eltype("running_mean", running_mean, S)
    check_bn_eltype("running_var", running_var, S)
    ps = bn_param_size(x)
    scale_p, bias_p = bn_param_array(scale, x), bn_param_array(bias, x)
    mean_p, var_p = bn_param_array(running_mean, x), bn_param_array(running_var, x)
    key = bn_inference_key(y, x, scale_p, bias_p, mean_p, var_p, deterministic,
                            math_mode, max_workspace)
    g = cached_graph(key) do
        build_batchnorm_inference_graph(y, x, scale_p, bias_p, mean_p, var_p;
                                         deterministic, math_mode, max_workspace)
    end
    invvar = similar(var_p, S, ps)
    @. invvar = 1 / sqrt(var_p + epsilon)
    execute!(g, tensor(g, "X")=>x, tensor(g, "Scale")=>scale_p,
             tensor(g, "Bias")=>bias_p, tensor(g, "Mean")=>mean_p,
             tensor(g, "InvVariance")=>invvar, tensor(g, "Y")=>y)
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
    alpha == 1 && beta == 0 && dalpha == 1 && dbeta == 0 ||
        throw(ArgumentError("batchnorm_gradient! requires alpha=1, beta=0, dalpha=1, and dbeta=0"))
    S = bn_stat_type(T)
    check_bn_eltype("scale", scale, S)
    check_bn_eltype("dscale", dscale, S)
    check_bn_eltype("dbias", dbias, S)
    check_bn_eltype("saved_mean", saved_mean, S)
    check_bn_eltype("saved_invvar", saved_invvar, S)
    scale_p = bn_param_array(scale, x)
    dscale_p, dbias_p = bn_param_array(dscale, x), bn_param_array(dbias, x)
    mean_p, invvar_p = bn_param_array(saved_mean, x), bn_param_array(saved_invvar, x)
    key = bn_gradient_key(dx, dscale_p, dbias_p, dy, x, scale_p, mean_p, invvar_p,
                           deterministic, math_mode, max_workspace)
    g = cached_graph(key) do
        build_batchnorm_gradient_graph(dx, dscale_p, dbias_p, dy, x, scale_p,
                                        mean_p, invvar_p; deterministic, math_mode,
                                        max_workspace)
    end
    execute!(g, tensor(g, "dY")=>dy, tensor(g, "X")=>x,
             tensor(g, "Scale")=>scale_p, tensor(g, "Mean")=>mean_p,
             tensor(g, "InvVariance")=>invvar_p, tensor(g, "dX")=>dx,
             tensor(g, "dScale")=>dscale_p, tensor(g, "dBias")=>dbias_p)
    return dx, dscale, dbias
end

function cached_graph_supported(f)
    try
        f()
        return true
    catch e
        graph_unsupported(e) || rethrow()
        return false
    end
end

function batchnorm_training_supported(y::DenseCuArray{T}, x::DenseCuArray{T},
                                      scale::DenseCuArray, bias::DenseCuArray;
                                      running_mean=nothing, running_var=nothing,
                                      momentum::Real=0.1, epsilon::Real=1e-5,
                                      deterministic::Bool=false,
                                      math_mode=CUDACore.math_mode(),
                                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    S = bn_stat_type(T)
    eltype(scale) == S && eltype(bias) == S || return false
    check_bn_running_stats(running_mean, running_var)
    running_mean === nothing || eltype(running_mean) == S || return false
    running_var === nothing || eltype(running_var) == S || return false
    epsilon = max(epsilon, CUDNN_BN_MIN_EPSILON)
    ps = bn_param_size(x)
    scale_p, bias_p = bn_param_array(scale, x), bn_param_array(bias, x)
    saved_mean = similar(scale, S, ps)
    saved_invvar = similar(scale, S, ps)
    rm_p = running_mean === nothing ? nothing : bn_param_array(running_mean, x)
    rv_p = running_var === nothing ? nothing : bn_param_array(running_var, x)
    key = bn_training_key(y, x, scale_p, bias_p, saved_mean, saved_invvar, rm_p, rv_p,
                           epsilon, momentum, deterministic, math_mode, max_workspace)
    return cached_graph_supported() do
        cached_graph(key) do
            build_batchnorm_training_graph(y, x, scale_p, bias_p, saved_mean,
                                            saved_invvar, rm_p, rv_p; deterministic,
                                            math_mode, max_workspace)
        end
    end
end

function batchnorm_inference_supported(y::DenseCuArray{T}, x::DenseCuArray{T},
                                       scale::DenseCuArray, bias::DenseCuArray,
                                       running_mean::DenseCuArray,
                                       running_var::DenseCuArray;
                                       deterministic::Bool=false,
                                       math_mode=CUDACore.math_mode(),
                                       max_workspace::Union{Nothing,Integer}=nothing) where {T}
    S = bn_stat_type(T)
    all(a -> eltype(a) == S, (scale, bias, running_mean, running_var)) || return false
    scale_p, bias_p = bn_param_array(scale, x), bn_param_array(bias, x)
    mean_p, var_p = bn_param_array(running_mean, x), bn_param_array(running_var, x)
    key = bn_inference_key(y, x, scale_p, bias_p, mean_p, var_p, deterministic,
                            math_mode, max_workspace)
    return cached_graph_supported() do
        cached_graph(key) do
            build_batchnorm_inference_graph(y, x, scale_p, bias_p, mean_p, var_p;
                                             deterministic, math_mode, max_workspace)
        end
    end
end

function batchnorm_gradient_supported(dx::DenseCuArray{T}, dscale::DenseCuArray,
                                      dbias::DenseCuArray, dy::DenseCuArray{T},
                                      x::DenseCuArray{T}, scale::DenseCuArray,
                                      saved_mean::DenseCuArray,
                                      saved_invvar::DenseCuArray;
                                      deterministic::Bool=false,
                                      math_mode=CUDACore.math_mode(),
                                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    S = bn_stat_type(T)
    all(a -> eltype(a) == S, (scale, dscale, dbias, saved_mean, saved_invvar)) ||
        return false
    scale_p = bn_param_array(scale, x)
    dscale_p, dbias_p = bn_param_array(dscale, x), bn_param_array(dbias, x)
    mean_p, invvar_p = bn_param_array(saved_mean, x), bn_param_array(saved_invvar, x)
    key = bn_gradient_key(dx, dscale_p, dbias_p, dy, x, scale_p, mean_p, invvar_p,
                           deterministic, math_mode, max_workspace)
    return cached_graph_supported() do
        cached_graph(key) do
            build_batchnorm_gradient_graph(dx, dscale_p, dbias_p, dy, x, scale_p,
                                            mean_p, invvar_p; deterministic, math_mode,
                                            max_workspace)
        end
    end
end

@doc raw"""
    batchnorm_training!(y, x, scale, bias; running_mean=nothing, running_var=nothing,
                        momentum=0.1, epsilon=1e-5) -> (saved_mean, saved_invvar)
    batchnorm_inference!(y, x, scale, bias, running_mean, running_var; epsilon=1e-5)
    batchnorm_gradient!(dx, dscale, dbias, dy, x, scale, saved_mean, saved_invvar;
                        epsilon=1e-5)
    batchnorm_training_supported(y, x, scale, bias; kwargs...) -> Bool
    batchnorm_inference_supported(y, x, scale, bias, running_mean, running_var;
                                  kwargs...) -> Bool
    batchnorm_gradient_supported(dx, dscale, dbias, dy, x, scale, saved_mean,
                                 saved_invvar; kwargs...) -> Bool

Spatial batch normalization, normalizing over all dimensions of `x` except the channel
dimension (the second-to-last). Parameters may be vectors of length `C` or shaped
`(1, ..., C, 1)`.

`Float16` and `BFloat16` inputs use `Float32` parameters and statistics. `Float32` and
`Float64` inputs use parameters and statistics of the same type.

Training returns the saved batch statistics that the gradient consumes, with
`saved_invvar = 1 / sqrt(var + epsilon)`, and updates `running_mean`/`running_var` in
place when given. Engine selection can be constrained with the `deterministic`,
`math_mode`, and `max_workspace` keywords.
"""
batchnorm_training!, batchnorm_inference!, batchnorm_gradient!,
batchnorm_training_supported, batchnorm_inference_supported,
batchnorm_gradient_supported

@public batchnorm_training!, batchnorm_inference!, batchnorm_gradient!,
        batchnorm_training_supported, batchnorm_inference_supported,
        batchnorm_gradient_supported
