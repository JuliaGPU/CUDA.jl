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

# The output size of PyTorch's and NNlib's `ceil_mode`, which lets the last window run past
# the end as long as it starts inside the input or the leading padding.
function pool_output_dim(x, window, pre, post, stride, ceil_mode)
    ceil_mode || return resample_output_dim(x, window, pre, post, stride)
    o = cld(x + pre + post - window, stride) + 1
    return (o - 1) * stride < x + pre ? o : o - 1
end

# Validate the pooling geometry, and resolve it into the padding the backends need: the
# caller's `pre` and `post`, which the fixed-function API pools with natively given the
# output size, and `graph_post`, which extends `post` to cover the last window for graphs.
function pool_geometry(y, x, window, stride, padding, ceil_mode)
    rank = ndims(x) - 2
    pre, post = convolution_padding(padding, rank)
    win = require_positive("pooling window", spatial_vector(window, rank))
    str = require_positive("pooling stride", spatial_vector(stride, rank))
    # this guarantees the output is at least as large as without `ceil_mode`, so that the
    # backends only ever need extra post padding
    ceil_mode && !all(0 .<= pre .< win .&& 0 .<= post .< win) &&
        throw(ArgumentError("ceil_mode requires non-negative padding smaller than the " *
                            "window, got padding $padding for window $window"))
    out = pool_output_dim.(size(x)[1:rank], win, pre, post, str, ceil_mode)
    expected = (out..., size(x, rank+1), size(x, rank+2))
    size(y) == expected ||
        throw(DimensionMismatch("pooling output has size $(size(y)), expected $expected"))
    graph_post = max.(post, (out .- 1) .* str .+ win .- size(x)[1:rank] .- pre)
    return win, str, pre, post, graph_post
end

function check_pool_gradient(dx, dy, y, x)
    size(dx) == size(x) ||
        throw(DimensionMismatch("gradient dx has size $(size(dx)), expected $(size(x))"))
    size(dy) == size(y) ||
        throw(DimensionMismatch("gradient dy has size $(size(dy)), expected $(size(y))"))
    return
end

# Counting the padding is the same as pooling a zero-padded input, which the mean-pooling
# functions use to avoid two limitations of graphs. With asymmetric padding, the graph API's
# resampling engine for NCHW tensors only counts padding up to `pre` elements past the end
# of the input (cuDNN 9.24-9.27, #3473), so make the padding symmetric by hand. A graph
# would also count the part of the last window that runs past the padding with `ceil_mode`,
# so then materialise all of the padding and exclude only that overhang.
# Returns the zero padding to apply, and the geometry to pool the padded input with.
function meanpool_padding(pre, post, graph_post, count_include_pad)
    none = zero(pre)
    if count_include_pad && graph_post != post && !use_fixed_pooling(pre, post)
        return pre, post, none, none, graph_post .- post, false
    elseif count_include_pad && pre != post
        pad = min.(pre, post)
        return pre .- pad, post .- pad, pad, pad, pad, true
    else
        return none, none, pre, post, graph_post, count_include_pad
    end
end

function fixed_pool!(y, x, mode, nan_propagation, window, padding, stride, alpha, beta)
    T = eltype(x)
    desc = fixed_pool_descriptor(mode, nan_propagation, window, padding, stride)
    cudnnPoolingForward(handle(), desc, scalingParameter(T, alpha),
                        strided_tensor_descriptor(x), x, scalingParameter(T, beta),
                        strided_tensor_descriptor(y), y)
    return y
end

function fixed_pool_bwd!(dx, dy, y, x, mode, nan_propagation, window, padding, stride,
                         alpha, beta)
    T = eltype(x)
    desc = fixed_pool_descriptor(mode, nan_propagation, window, padding, stride)
    cudnnPoolingBackward(handle(), desc, scalingParameter(T, alpha),
                         strided_tensor_descriptor(y), y, strided_tensor_descriptor(dy), dy,
                         strided_tensor_descriptor(x), x, scalingParameter(T, beta),
                         strided_tensor_descriptor(dx), dx)
    return dx
end

function pool_forward!(y, x, op, count_include_pad, win, str, pre, post, graph_post,
                       alpha, beta, nan_propagation, deterministic, math_mode,
                       max_workspace)
    x = cudnn_input(x)
    cudnn_output(y, beta) do y
        if use_fixed_pooling(pre, post)
            mode = fixed_pool_mode(op, count_include_pad, deterministic)
            fixed_pool!(y, x, mode, nan_propagation, win, pre, str, alpha, beta)
        else
            graph_mode, padding_mode = graph_pool_mode(op, count_include_pad)
            pool_execute!(y, x, op, graph_mode, padding_mode, win, pre, graph_post, str,
                          nan_propagation, alpha, beta, deterministic, math_mode,
                          max_workspace)
        end
    end
end

function pool_backward!(dx, dy, y, x, op, count_include_pad, win, str, pre, post,
                        graph_post, alpha, beta, nan_propagation, deterministic, math_mode,
                        max_workspace; xlayout=describable, ylayout=describable)
    x, y, dy = cudnn_input(x, xlayout), cudnn_input(y, ylayout), cudnn_input(dy, ylayout)
    cudnn_output(dx, beta, xlayout) do dx
        if use_fixed_pooling(pre, post)
            mode = fixed_pool_mode(op, count_include_pad, deterministic)
            fixed_pool_bwd!(dx, dy, y, x, mode, nan_propagation, win, pre, str, alpha,
                            beta)
        else
            graph_mode, padding_mode = graph_pool_mode(op, count_include_pad)
            pool_bwd_execute!(dx, dy, y, x, Symbol(op, :_bwd), graph_mode, padding_mode,
                              win, pre, graph_post, str, nan_propagation, alpha, beta,
                              deterministic, math_mode, max_workspace)
        end
    end
end

graph_pool_mode(op, count_include_pad) =
    op === :maxpool ? (:maxpool, :neg_inf) :
    count_include_pad ? (:avgpool_include_padding, :zero) :
                        (:avgpool_exclude_padding, :zero)

function maxpool!(y::StridedCuArray, x::StridedCuArray; window, stride=window, padding=0,
                  ceil_mode::Bool=false, alpha::Real=1, beta::Real=0,
                  nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                  deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                  max_workspace::Union{Nothing,Integer}=nothing)
    win, str, pre, post, graph_post = pool_geometry(y, x, window, stride, padding,
                                                    ceil_mode)
    isempty(y) && return y
    pool_forward!(y, x, :maxpool, false, win, str, pre, post, graph_post, alpha, beta,
                  nan_propagation, deterministic, math_mode, max_workspace)
end

function meanpool!(y::StridedCuArray, x::StridedCuArray; window, stride=window, padding=0,
                   ceil_mode::Bool=false, count_include_pad::Bool=true, alpha::Real=1,
                   beta::Real=0, nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                   deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                   max_workspace::Union{Nothing,Integer}=nothing)
    win, str, pre, post, graph_post = pool_geometry(y, x, window, stride, padding,
                                                    ceil_mode)
    isempty(y) && return y
    lo, hi, pre, post, graph_post, include_pad =
        meanpool_padding(pre, post, graph_post, count_include_pad)
    any(!iszero, (lo..., hi...)) && (x = zero_padded(x, lo, hi))
    pool_forward!(y, x, :meanpool, include_pad, win, str, pre, post, graph_post, alpha,
                  beta, nan_propagation, deterministic, math_mode, max_workspace)
end

# The strides cuDNN describes an operand with once `cudnn_input` or `cudnn_output` stages it.
staged_strides(a) = describable(a) ? view_strides(a) : collect(tuple_strides(size(a)))

# The max-pooling gradient needs `dx` laid out like `x`, and `dy` like `y`: the fixed-function
# API rejects other layouts, and the graph API computes wrong gradients for them (#3497).
# Stage both members of a mismatched pair densely, which reuses one that already is.
pair_layout(a, b) = staged_strides(a) == staged_strides(b) ? describable : dense_layout

function ∇maxpool!(dx::StridedCuArray{T}, dy::StridedCuArray{T}, y::StridedCuArray{T},
                   x::StridedCuArray{T}; window, stride=window, padding=0,
                   ceil_mode::Bool=false, alpha::Real=1, beta::Real=0,
                   nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                   deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                   max_workspace::Union{Nothing,Integer}=nothing) where {T}
    win, str, pre, post, graph_post = pool_geometry(y, x, window, stride, padding,
                                                    ceil_mode)
    check_pool_gradient(dx, dy, y, x)
    isempty(dx) && return dx
    pool_backward!(dx, dy, y, x, :maxpool, false, win, str, pre, post, graph_post, alpha,
                   beta, nan_propagation, deterministic, math_mode, max_workspace;
                   xlayout=pair_layout(x, dx), ylayout=pair_layout(y, dy))
end

function ∇meanpool!(dx::StridedCuArray{T}, dy::StridedCuArray{T}, y::StridedCuArray{T},
                    x::StridedCuArray{T}; window, stride=window, padding=0,
                    ceil_mode::Bool=false, count_include_pad::Bool=true, alpha::Real=1,
                    beta::Real=0, nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                    deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                    max_workspace::Union{Nothing,Integer}=nothing) where {T}
    win, str, pre, post, graph_post = pool_geometry(y, x, window, stride, padding,
                                                    ceil_mode)
    check_pool_gradient(dx, dy, y, x)
    isempty(dx) && return dx
    lo, hi, pre, post, graph_post, include_pad =
        meanpool_padding(pre, post, graph_post, count_include_pad)
    if all(iszero, (lo..., hi...))
        return pool_backward!(dx, dy, y, x, :meanpool, include_pad, win, str, pre, post,
                              graph_post, alpha, beta, nan_propagation, deterministic,
                              math_mode, max_workspace)
    end
    padded_x = zero_padded(x, lo, hi)
    dx_padded = similar(padded_x)
    pool_backward!(dx_padded, dy, y, padded_x, :meanpool, include_pad, win, str, pre, post,
                   graph_post, alpha, 0, nan_propagation, deterministic, math_mode,
                   max_workspace)
    _, ranges = zero_padded_geometry(x, lo, hi)
    interior = view(dx_padded, ranges...)
    beta == 0 ? copyto!(dx, interior) : (dx .= interior .+ beta .* dx)
    return dx
end

@doc raw"""
    maxpool!(y, x; window, stride=window, padding=0, ceil_mode=false, alpha=1, beta=0,
             nan_propagation=CUDNN_PROPAGATE_NAN)
    meanpool!(y, x; window, stride=window, padding=0, ceil_mode=false,
              count_include_pad=true, alpha=1, beta=0, nan_propagation=CUDNN_PROPAGATE_NAN)
    ∇maxpool!(dx, dy, y, x; kwargs...)
    ∇meanpool!(dx, dy, y, x; kwargs...)

Pool over the spatial dimensions of `x` (Julia memory order: spatial dimensions first,
then channels, then batch), or compute the gradient with respect to the pooling input
given the forward input `x` and output `y`. The arrays may be strided views, such as a range
of channels. Views that cuDNN can't address, such as reversed ones, are copied, as are
the arrays of a max-pooling gradient whose layout differs from the forward array's.

`padding` accepts a scalar, one value per spatial dimension, or per-side
`(pre1, post1, ...)` pairs. `count_include_pad` includes padded elements in the averaging
divisor. `nan_propagation=CUDNN_NOT_PROPAGATE_NAN` makes max pooling ignore NaNs.

`y` must have the output size of the pooling geometry. With `ceil_mode=true`, as in
PyTorch, that size is rounded up so that the last window may run past the end of the input
and its padding, as long as it starts inside the input or the leading padding; the
averaging divisor never counts that overhang. This requires the padding to be smaller than
the window.

Pooling with symmetric padding uses cuDNN's fixed-function API, which is faster than the
graph API; asymmetric padding needs a graph, except for mean pooling with
`count_include_pad`, which first copies `x` into a zero-padded array. `deterministic=true` avoids the atomics of
the regular max-pooling gradient. Engine selection for graphs can be further constrained
with the `math_mode` and `max_workspace` keywords.
"""
maxpool!, meanpool!, ∇maxpool!, ∇meanpool!

@public maxpool!, meanpool!, ∇maxpool!, ∇meanpool!
