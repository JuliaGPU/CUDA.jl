function pool_key(op, y, x, mode, padding_mode, window, pre_padding, post_padding, stride,
                   nan_propagation, alpha, beta, deterministic, math_mode, max_workspace)
    (op, eltype(x),
     size(x), strides(x), pointer_alignment(x),
     size(y), strides(y), pointer_alignment(y),
     mode, padding_mode, Tuple(window), Tuple(pre_padding), Tuple(post_padding),
     Tuple(stride), nan_propagation, Float64(alpha), Float64(beta), deterministic,
     math_mode, max_workspace)
end

function pool_bwd_key(op, dx, dy, y, x, mode, padding_mode, window, pre_padding,
                       post_padding, stride, nan_propagation, alpha, beta, deterministic,
                       math_mode, max_workspace)
    (op, eltype(x),
     size(x), strides(x), pointer_alignment(x),
     size(y), strides(y), pointer_alignment(y),
     size(dy), strides(dy), pointer_alignment(dy),
     size(dx), strides(dx), pointer_alignment(dx),
     mode, padding_mode, Tuple(window), Tuple(pre_padding), Tuple(post_padding),
     Tuple(stride), nan_propagation, Float64(alpha), Float64(beta), deterministic,
     math_mode, max_workspace)
end

function build_pool_graph(y, x, mode, padding_mode, window, pre_padding, post_padding,
                           stride, nan_propagation, alpha, beta;
                           deterministic, math_mode, max_workspace)
    ctype = conv_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tx = tensor!(g, x; name="X")
    ty = tensor!(g, y; name="Y", output=true)
    resample_fwd!(g, tx; y=ty, mode, padding_mode, window, pre_padding, post_padding,
                  stride, nan_propagation, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

function build_pool_bwd_graph(dx, dy, y, x, mode, padding_mode, window, pre_padding,
                               post_padding, stride, nan_propagation, alpha, beta;
                               deterministic, math_mode, max_workspace)
    ctype = conv_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tx = tensor!(g, x; name="X")
    ty = tensor!(g, y; name="Y")
    tdy = tensor!(g, dy; name="dY")
    tdx = tensor!(g, dx; name="dX", output=true)
    resample_bwd!(g, tdy; dx=tdx, x=tx, y=ty, mode, padding_mode, window, pre_padding,
                  post_padding, stride, nan_propagation, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

function pool_execute!(y, x, op, mode, padding_mode, window, pre_padding, post_padding,
                        stride, nan_propagation, alpha, beta, deterministic, math_mode,
                        max_workspace)
    key = pool_key(op, y, x, mode, padding_mode, window, pre_padding, post_padding, stride,
                    nan_propagation, alpha, beta, deterministic, math_mode, max_workspace)
    g = cached_graph(key) do
        build_pool_graph(y, x, mode, padding_mode, window, pre_padding, post_padding, stride,
                          nan_propagation, alpha, beta; deterministic, math_mode,
                          max_workspace)
    end
    execute!(g, tensor(g, "X")=>x, tensor(g, "Y")=>y)
    return y
end

function pool_bwd_execute!(dx, dy, y, x, op, mode, padding_mode, window, pre_padding,
                            post_padding, stride, nan_propagation, alpha, beta,
                            deterministic, math_mode, max_workspace)
    key = pool_bwd_key(op, dx, dy, y, x, mode, padding_mode, window, pre_padding,
                        post_padding, stride, nan_propagation, alpha, beta, deterministic,
                        math_mode, max_workspace)
    g = cached_graph(key) do
        build_pool_bwd_graph(dx, dy, y, x, mode, padding_mode, window, pre_padding,
                              post_padding, stride, nan_propagation, alpha, beta;
                              deterministic, math_mode, max_workspace)
    end
    execute!(g, tensor(g, "X")=>x, tensor(g, "Y")=>y, tensor(g, "dY")=>dy,
             tensor(g, "dX")=>dx)
    return dx
end

# The fixed-function API is faster than the graph API's single resampling engine (up to
# twice as fast for the Float16 max-pooling gradient with cuDNN 9.26), and supports 1D
# pooling, but only symmetric padding.
use_fixed_pooling(pre, post) = pre == post && length(pre) <= 3

function fixed_pool_mode(op, count_include_pad, deterministic)
    if op === :maxpool
        # the regular max-pooling gradient uses atomics
        deterministic ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_MAX
    else
        count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
                            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    end
end

function fixed_pool_descriptor(mode, nan_propagation, window, padding, stride)
    # cuDNN orders spatial dimensions slowest-varying first, and pools at least two;
    # `dim4` gives the tensors of 1D pooling a trailing singleton dimension.
    cudnn_order(v, extra) = Cint[reverse(v); length(v) == 1 ? [extra] : Int[]]
    win = cudnn_order(window, 1)
    cudnnPoolingDescriptor(mode, nan_propagation, Cint(length(win)), win,
                           cudnn_order(padding, 0), cudnn_order(stride, 1))
end

function check_pool_output(y, x, window, pre, post, stride)
    rank = ndims(x) - 2
    expected = (resample_output_dims(size(x)[1:rank], window, pre, post, stride)...,
                size(x, rank+1), size(x, rank+2))
    size(y) == expected ||
        throw(DimensionMismatch("pooling output has size $(size(y)), expected $expected"))
    return
end

function fixed_pool!(y, x, mode, nan_propagation, window, padding, stride, alpha, beta)
    T = eltype(x)
    desc = fixed_pool_descriptor(mode, nan_propagation, window, padding, stride)
    cudnnPoolingForward(handle(), desc, scalingParameter(T, alpha),
                        cudnnTensorDescriptor(x), x, scalingParameter(T, beta),
                        cudnnTensorDescriptor(y), y)
    return y
end

function fixed_pool_bwd!(dx, dy, y, x, mode, nan_propagation, window, padding, stride,
                         alpha, beta)
    T = eltype(x)
    desc = fixed_pool_descriptor(mode, nan_propagation, window, padding, stride)
    cudnnPoolingBackward(handle(), desc, scalingParameter(T, alpha),
                         cudnnTensorDescriptor(y), y, cudnnTensorDescriptor(dy), dy,
                         cudnnTensorDescriptor(x), x, scalingParameter(T, beta),
                         cudnnTensorDescriptor(dx), dx)
    return dx
end

function maxpool!(y::DenseCuArray, x::DenseCuArray; window, stride=window, padding=0,
                  alpha::Real=1, beta::Real=0,
                  nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                  deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                  max_workspace::Union{Nothing,Integer}=nothing)
    isempty(y) && return y
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    check_pool_output(y, x, win, pre, post, str)
    if use_fixed_pooling(pre, post)
        mode = fixed_pool_mode(:maxpool, false, deterministic)
        fixed_pool!(y, x, mode, nan_propagation, win, pre, str, alpha, beta)
    else
        pool_execute!(y, x, :maxpool, :maxpool, :neg_inf, win, pre, post, str,
                      nan_propagation, alpha, beta, deterministic, math_mode,
                      max_workspace)
    end
end

function meanpool!(y::DenseCuArray, x::DenseCuArray; window, stride=window, padding=0,
                   count_include_pad::Bool=true, alpha::Real=1, beta::Real=0,
                   nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                   deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                   max_workspace::Union{Nothing,Integer}=nothing)
    isempty(y) && return y
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    check_pool_output(y, x, win, pre, post, str)
    if use_fixed_pooling(pre, post)
        mode = fixed_pool_mode(:meanpool, count_include_pad, deterministic)
        fixed_pool!(y, x, mode, nan_propagation, win, pre, str, alpha, beta)
    else
        graph_mode = count_include_pad ? :avgpool_include_padding :
                                         :avgpool_exclude_padding
        pool_execute!(y, x, :meanpool, graph_mode, :zero, win, pre, post, str,
                      nan_propagation, alpha, beta, deterministic, math_mode,
                      max_workspace)
    end
end

function ∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T},
                   x::DenseCuArray{T}; window, stride=window, padding=0,
                   alpha::Real=1, beta::Real=0,
                   nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                   deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                   max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(dx) && return dx
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    check_pool_output(y, x, win, pre, post, str)
    if use_fixed_pooling(pre, post)
        mode = fixed_pool_mode(:maxpool, false, deterministic)
        fixed_pool_bwd!(dx, dy, y, x, mode, nan_propagation, win, pre, str, alpha, beta)
    else
        pool_bwd_execute!(dx, dy, y, x, :maxpool_bwd, :maxpool, :neg_inf, win,
                          pre, post, str, nan_propagation, alpha, beta, deterministic,
                          math_mode, max_workspace)
    end
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T},
                    x::DenseCuArray{T}; window, stride=window, padding=0,
                    count_include_pad::Bool=true, alpha::Real=1, beta::Real=0,
                    nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                    deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                    max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(dx) && return dx
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    check_pool_output(y, x, win, pre, post, str)
    if use_fixed_pooling(pre, post)
        mode = fixed_pool_mode(:meanpool, count_include_pad, deterministic)
        fixed_pool_bwd!(dx, dy, y, x, mode, nan_propagation, win, pre, str, alpha, beta)
    else
        graph_mode = count_include_pad ? :avgpool_include_padding :
                                         :avgpool_exclude_padding
        pool_bwd_execute!(dx, dy, y, x, :meanpool_bwd, graph_mode, :zero, win,
                          pre, post, str, nan_propagation, alpha, beta, deterministic,
                          math_mode, max_workspace)
    end
end

@doc raw"""
    maxpool!(y, x; window, stride=window, padding=0, alpha=1, beta=0,
             nan_propagation=CUDNN_PROPAGATE_NAN)
    meanpool!(y, x; window, stride=window, padding=0, count_include_pad=true,
              alpha=1, beta=0, nan_propagation=CUDNN_PROPAGATE_NAN)
    ∇maxpool!(dx, dy, y, x; kwargs...)
    ∇meanpool!(dx, dy, y, x; kwargs...)

Pool over the spatial dimensions of `x` (Julia memory order: spatial dimensions first,
then channels, then batch), or compute the gradient with respect to the pooling input
given the forward input `x` and output `y`.

`padding` accepts a scalar, one value per spatial dimension, or per-side
`(pre1, post1, ...)` pairs. `count_include_pad` includes padded elements in the averaging
divisor. `nan_propagation=CUDNN_NOT_PROPAGATE_NAN` makes max pooling ignore NaNs.

Pooling with symmetric padding uses cuDNN's fixed-function API, which is faster than the
graph API; asymmetric padding needs a graph. `deterministic=true` avoids the atomics of
the regular max-pooling gradient. Engine selection for graphs can be further constrained
with the `math_mode` and `max_workspace` keywords.
"""
maxpool!, meanpool!, ∇maxpool!, ∇meanpool!

@public maxpool!, meanpool!, ∇maxpool!, ∇meanpool!
