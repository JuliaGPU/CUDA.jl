function pool_key(op, y, x, mode, padding_mode, window, pre_padding, post_padding, stride,
                   alpha, beta, deterministic, math_mode, max_workspace)
    (op, eltype(x),
     size(x), strides(x), pointer_alignment(x),
     size(y), strides(y), pointer_alignment(y),
     mode, padding_mode, Tuple(window), Tuple(pre_padding), Tuple(post_padding),
     Tuple(stride), Float64(alpha), Float64(beta), deterministic, math_mode, max_workspace)
end

function pool_bwd_key(op, dx, dy, y, x, mode, padding_mode, window, pre_padding,
                       post_padding, stride, alpha, beta, deterministic, math_mode,
                       max_workspace)
    (op, eltype(x),
     size(x), strides(x), pointer_alignment(x),
     size(y), strides(y), pointer_alignment(y),
     size(dy), strides(dy), pointer_alignment(dy),
     size(dx), strides(dx), pointer_alignment(dx),
     mode, padding_mode, Tuple(window), Tuple(pre_padding), Tuple(post_padding),
     Tuple(stride), Float64(alpha), Float64(beta), deterministic, math_mode, max_workspace)
end

function build_pool_graph(y, x, mode, padding_mode, window, pre_padding, post_padding,
                           stride, alpha, beta; deterministic, math_mode, max_workspace)
    ctype = conv_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tx = tensor!(g, x; name="X")
    ty = tensor!(g, y; name="Y", output=true)
    resample_fwd!(g, tx; y=ty, mode, padding_mode, window, pre_padding, post_padding,
                  stride, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

function build_pool_bwd_graph(dx, dy, y, x, mode, padding_mode, window, pre_padding,
                               post_padding, stride, alpha, beta;
                               deterministic, math_mode, max_workspace)
    ctype = conv_compute_type(eltype(x))
    g = Graph(io_dtype=eltype(x), intermediate_dtype=ctype, compute_dtype=ctype)
    tx = tensor!(g, x; name="X")
    ty = tensor!(g, y; name="Y")
    tdy = tensor!(g, dy; name="dY")
    tdx = tensor!(g, dx; name="dX", output=true)
    resample_bwd!(g, tdy; dx=tdx, x=tx, y=ty, mode, padding_mode, window, pre_padding,
                  post_padding, stride, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

function pool_execute!(y, x, op, mode, padding_mode, window, pre_padding, post_padding,
                        stride, alpha, beta, deterministic, math_mode, max_workspace)
    key = pool_key(op, y, x, mode, padding_mode, window, pre_padding, post_padding, stride,
                    alpha, beta, deterministic, math_mode, max_workspace)
    g = cached_graph(key) do
        build_pool_graph(y, x, mode, padding_mode, window, pre_padding, post_padding, stride,
                          alpha, beta; deterministic, math_mode, max_workspace)
    end
    execute!(g, tensor(g, "X")=>x, tensor(g, "Y")=>y)
    return y
end

function pool_bwd_execute!(dx, dy, y, x, op, mode, padding_mode, window, pre_padding,
                            post_padding, stride, alpha, beta, deterministic, math_mode,
                            max_workspace)
    key = pool_bwd_key(op, dx, dy, y, x, mode, padding_mode, window, pre_padding,
                        post_padding, stride, alpha, beta, deterministic, math_mode,
                        max_workspace)
    g = cached_graph(key) do
        build_pool_bwd_graph(dx, dy, y, x, mode, padding_mode, window, pre_padding,
                              post_padding, stride, alpha, beta; deterministic, math_mode,
                              max_workspace)
    end
    execute!(g, tensor(g, "X")=>x, tensor(g, "Y")=>y, tensor(g, "dY")=>dy,
             tensor(g, "dX")=>dx)
    return dx
end

# Convert the integer, tuple or array to pooling dims compatible with array size
function pooldims(d, s::Dims{N}) where N
    if d isa Integer || length(d) == N-2
        Cint[reverse(min.(d,s[1:N-2]))...]
    else
        throw(DimensionMismatch("Cannot pool $(Base.dims2string(s)) array with $d pooldims."))
    end
end

pooldims(d, s::Dims{3}) = pooldims(d, (1,s...))
pooldims(d, s::Dims{2}) = pooldims(d, (1,1,s...))
pooldims(d, s::Dims{1}) = pooldims(d, (1,1,1,s...))
pooldims(d, s::Dims{0}) = pooldims(d, (1,1,1,1))


function legacy_pooling_forward!(y::DenseCuArray{T}, x::DenseCuArray{T};
                                  mode, window, padding, stride, alpha, beta) where {T}
    desc = cudnnPoolingDescriptor(mode, CUDNN_NOT_PROPAGATE_NAN,
                                  Cint(max(2, ndims(x)-2)), pooldims(window, size(x)),
                                  pooldims(padding, size(x)), pooldims(stride, size(x)))
    xdesc, ydesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(y)
    alpha, beta = scalingParameter(T, alpha), scalingParameter(T, beta)
    cudnnPoolingForward(handle(), desc, alpha, xdesc, x, beta, ydesc, y)
    return y
end

function legacy_pooling_backward!(dx::DenseCuArray{T}, dy::DenseCuArray{T},
                                   y::DenseCuArray{T}, x::DenseCuArray{T};
                                   mode, window, padding, stride, alpha, beta) where {T}
    desc = cudnnPoolingDescriptor(mode, CUDNN_NOT_PROPAGATE_NAN,
                                  Cint(max(2, ndims(x)-2)), pooldims(window, size(x)),
                                  pooldims(padding, size(x)), pooldims(stride, size(x)))
    xdesc, ydesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(y)
    alpha, beta = scalingParameter(T, alpha), scalingParameter(T, beta)
    cudnnPoolingBackward(handle(), desc, alpha, ydesc, y, ydesc, dy, xdesc, x, beta,
                         xdesc, dx)
    return dx
end

function maxpool!(y::DenseCuArray, x::DenseCuArray; window, stride=window, padding=0,
                  alpha::Real=1, beta::Real=0, deterministic::Bool=false,
                  math_mode=CUDACore.math_mode(),
                  max_workspace::Union{Nothing,Integer}=nothing)
    isempty(y) && return y
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    try
        return pool_execute!(y, x, :maxpool, :maxpool, :neg_inf, win, pre, post, str,
                              alpha, beta, deterministic, math_mode, max_workspace)
    catch e
        graph_unsupported(e) || rethrow()
        pre == post || rethrow()
    end
    pad = symmetric_padding(padding, spatial_rank)
    legacy_pooling_forward!(y, x; mode=CUDNN_POOLING_MAX, window=win, padding=pad,
                             stride=str, alpha, beta)
end

function meanpool!(y::DenseCuArray, x::DenseCuArray; window, stride=window, padding=0,
                   count_include_pad::Bool=true, alpha::Real=1, beta::Real=0,
                   deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                   max_workspace::Union{Nothing,Integer}=nothing)
    isempty(y) && return y
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    graph_mode = count_include_pad ? :avgpool_include_padding : :avgpool_exclude_padding
    try
        return pool_execute!(y, x, :meanpool, graph_mode, :zero, win, pre, post, str,
                              alpha, beta, deterministic, math_mode, max_workspace)
    catch e
        graph_unsupported(e) || rethrow()
        pre == post || rethrow()
    end
    mode = count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
                               CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    pad = symmetric_padding(padding, spatial_rank)
    legacy_pooling_forward!(y, x; mode, window=win, padding=pad, stride=str, alpha, beta)
end

function ∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T},
                   x::DenseCuArray{T}; window, stride=window, padding=0,
                   alpha::Real=1, beta::Real=0, deterministic::Bool=false,
                   math_mode=CUDACore.math_mode(),
                   max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(dx) && return dx
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    try
        return pool_bwd_execute!(dx, dy, y, x, :maxpool_bwd, :maxpool, :neg_inf, win,
                                  pre, post, str, alpha, beta, deterministic, math_mode,
                                  max_workspace)
    catch e
        graph_unsupported(e) || rethrow()
        pre == post || rethrow()
    end
    pad = symmetric_padding(padding, spatial_rank)
    legacy_pooling_backward!(dx, dy, y, x; mode=CUDNN_POOLING_MAX, window=win,
                              padding=pad, stride=str, alpha, beta)
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T},
                    x::DenseCuArray{T}; window, stride=window, padding=0,
                    count_include_pad::Bool=true, alpha::Real=1, beta::Real=0,
                    deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                    max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(dx) && return dx
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    graph_mode = count_include_pad ? :avgpool_include_padding : :avgpool_exclude_padding
    try
        return pool_bwd_execute!(dx, dy, y, x, :meanpool_bwd, graph_mode, :zero, win,
                                  pre, post, str, alpha, beta, deterministic, math_mode,
                                  max_workspace)
    catch e
        graph_unsupported(e) || rethrow()
        pre == post || rethrow()
    end
    mode = count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
                               CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    pad = symmetric_padding(padding, spatial_rank)
    legacy_pooling_backward!(dx, dy, y, x; mode, window=win, padding=pad, stride=str,
                              alpha, beta)
end

@doc raw"""
    maxpool!(y, x; window, stride=window, padding=0, alpha=1, beta=0)
    meanpool!(y, x; window, stride=window, padding=0, count_include_pad=true,
              alpha=1, beta=0)
    ∇maxpool!(dx, dy, y, x; kwargs...)
    ∇meanpool!(dx, dy, y, x; kwargs...)

Pool over the spatial dimensions of `x` (Julia memory order: spatial dimensions first,
then channels, then batch), or compute the gradient with respect to the pooling input
given the forward input `x` and output `y`.

`padding` accepts a scalar, one value per spatial dimension, or per-side
`(pre1, post1, ...)` pairs. `count_include_pad` includes padded elements in the averaging
divisor. Engine selection can be constrained with the `deterministic`, `math_mode`, and
`max_workspace` keywords.
"""
maxpool!, meanpool!, ∇maxpool!, ∇meanpool!

@public maxpool!, meanpool!, ∇maxpool!, ∇meanpool!
