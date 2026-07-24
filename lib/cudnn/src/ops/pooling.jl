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

function maxpool!(y::DenseCuArray, x::DenseCuArray; window, stride=window, padding=0,
                  alpha::Real=1, beta::Real=0, deterministic::Bool=false,
                  math_mode=CUDACore.math_mode(),
                  max_workspace::Union{Nothing,Integer}=nothing)
    isempty(y) && return y
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    win = spatial_vector(window, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    pool_execute!(y, x, :maxpool, :maxpool, :neg_inf, win, pre, post, str,
                  alpha, beta, deterministic, math_mode, max_workspace)
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
    pool_execute!(y, x, :meanpool, graph_mode, :zero, win, pre, post, str,
                  alpha, beta, deterministic, math_mode, max_workspace)
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
    pool_bwd_execute!(dx, dy, y, x, :maxpool_bwd, :maxpool, :neg_inf, win,
                      pre, post, str, alpha, beta, deterministic, math_mode,
                      max_workspace)
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
    pool_bwd_execute!(dx, dy, y, x, :meanpool_bwd, graph_mode, :zero, win,
                      pre, post, str, alpha, beta, deterministic, math_mode,
                      max_workspace)
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
