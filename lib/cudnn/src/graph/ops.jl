struct SDPAMaskSubgraph
    input::Tensor
    fill::Tensor
    output::Tensor
end

struct SDPAFwdOp <: Operation
    q::Tensor
    k::Tensor
    v::Tensor
    o::Tensor
    scale::Tensor
    stats::Union{Nothing,Tensor}
    seq_len_q::Union{Nothing,Tensor}
    seq_len_kv::Union{Nothing,Tensor}
    mask_subgraph::Union{Nothing,SDPAMaskSubgraph}
end

struct SDPABwdOp <: Operation
    q::Tensor
    k::Tensor
    v::Tensor
    o::Tensor
    dO::Tensor
    stats::Tensor
    scale::Tensor
    dQ::Tensor
    dK::Tensor
    dV::Tensor
    seq_len_q::Union{Nothing,Tensor}
    seq_len_kv::Union{Nothing,Tensor}
    mask_subgraph::Union{Nothing,SDPAMaskSubgraph}
end

struct PointwiseOp <: Operation
    mode::cudnnPointwiseMode_t
    x::Tensor
    y::Tensor
    b::Union{Nothing,Tensor}
    t::Union{Nothing,Tensor}
    compute_dtype::cudnnDataType_t
    nan_propagation::cudnnNanPropagation_t
    relu_lower_clip::Float64
    relu_upper_clip::Float64
    relu_lower_clip_slope::Float64
    elu_alpha::Float64
    softplus_beta::Float64
    swish_beta::Float64
    axis::Union{Nothing,Int64}
    alpha1::Float32
    alpha2::Float32
end

struct MatmulOp <: Operation
    a::Tensor
    b::Tensor
    c::Tensor
    compute_dtype::cudnnDataType_t
end

struct ReductionOp <: Operation
    mode::cudnnReduceTensorOp_t
    x::Tensor
    y::Tensor
    compute_dtype::cudnnDataType_t
    deterministic::Bool
end

struct ConvFpropOp <: Operation
    x::Tensor
    w::Tensor
    y::Tensor
    pre_padding::Vector{Int64}
    post_padding::Vector{Int64}
    stride::Vector{Int64}
    dilation::Vector{Int64}
    mode::cudnnConvolutionMode_t
    compute_dtype::cudnnDataType_t
    alpha::Float64
    beta::Float64
end

struct ConvDgradOp <: Operation
    dy::Tensor
    w::Tensor
    dx::Tensor
    pre_padding::Vector{Int64}
    post_padding::Vector{Int64}
    stride::Vector{Int64}
    dilation::Vector{Int64}
    mode::cudnnConvolutionMode_t
    compute_dtype::cudnnDataType_t
    alpha::Float64
    beta::Float64
end

struct ConvWgradOp <: Operation
    x::Tensor
    dy::Tensor
    dw::Tensor
    pre_padding::Vector{Int64}
    post_padding::Vector{Int64}
    stride::Vector{Int64}
    dilation::Vector{Int64}
    mode::cudnnConvolutionMode_t
    compute_dtype::cudnnDataType_t
    alpha::Float64
    beta::Float64
end

struct ResampleFwdOp <: Operation
    x::Tensor
    y::Tensor
    index::Union{Nothing,Tensor}
    mode::cudnnResampleMode_t
    padding_mode::cudnnPaddingMode_t
    window::Vector{Int64}
    pre_padding::Vector{Int64}
    post_padding::Vector{Int64}
    stride::Vector{Int64}
    compute_dtype::cudnnDataType_t
    nan_propagation::cudnnNanPropagation_t
    alpha::Float64
    beta::Float64
end

struct ResampleBwdOp <: Operation
    dx::Tensor
    dy::Tensor
    x::Union{Nothing,Tensor}
    y::Union{Nothing,Tensor}
    index::Union{Nothing,Tensor}
    mode::cudnnResampleMode_t
    padding_mode::cudnnPaddingMode_t
    window::Vector{Int64}
    pre_padding::Vector{Int64}
    post_padding::Vector{Int64}
    stride::Vector{Int64}
    compute_dtype::cudnnDataType_t
    nan_propagation::cudnnNanPropagation_t
    alpha::Float64
    beta::Float64
end

struct NormFwdOp <: Operation
    x::Tensor
    mean::Union{Nothing,Tensor}
    inv_variance::Union{Nothing,Tensor}
    scale::Tensor
    bias::Tensor
    epsilon::Union{Nothing,Tensor}
    momentum::Union{Nothing,Tensor}
    input_running_mean::Union{Nothing,Tensor}
    input_running_var::Union{Nothing,Tensor}
    output_running_mean::Union{Nothing,Tensor}
    output_running_var::Union{Nothing,Tensor}
    y::Tensor
    mode::cudnnBackendNormMode_t
    phase::cudnnBackendNormFwdPhase_t
end

struct NormBwdOp <: Operation
    dy::Tensor
    x::Tensor
    scale::Tensor
    mean::Tensor
    inv_variance::Tensor
    dx::Tensor
    dscale::Tensor
    dbias::Tensor
    mode::cudnnBackendNormMode_t
end

const SDPA_BACKEND_ORDER = (4, 2, 3, 1)

function sdpa_tensor!(t::Tensor)
    length(t.dims) == 4 || throw(ArgumentError("SDPA tensors must be rank 4"))
    default_order = [4, 3, 2, 1]
    order = collect(SDPA_BACKEND_ORDER)
    t.backend_order == default_order || t.backend_order == order ||
        throw(ArgumentError("tensor $(t.name) has an incompatible dimension order for SDPA"))
    t.backend_order = order
    return t
end

function check_sdpa_sequence_lengths(seq_len_q, seq_len_kv, b)
    (seq_len_q === nothing) == (seq_len_kv === nothing) ||
        throw(ArgumentError("seq_len_q and seq_len_kv must be passed together"))
    seq_len_q === nothing && return
    seq_len_q.dims == [1, 1, 1, b] ||
        throw(DimensionMismatch("seq_len_q dimensions must be $((1, 1, 1, b))"))
    seq_len_kv.dims == [1, 1, 1, b] ||
        throw(DimensionMismatch("seq_len_kv dimensions must be $((1, 1, 1, b))"))
    seq_len_q.dtype == CUDNN_DATA_INT32 ||
        throw(ArgumentError("seq_len_q must have Int32 dtype"))
    seq_len_kv.dtype == CUDNN_DATA_INT32 ||
        throw(ArgumentError("seq_len_kv must have Int32 dtype"))
    return
end

function sdpa_causal_subgraph!(g::Graph, skv, sq, hq, b)
    score_dims = (skv, sq, hq, b)
    input = tensor!(g; dims=score_dims, dtype=Float32, virtual=true, name="Score")
    fill = scalar!(g, Float32; rank=4, name="MaskValue")
    output = tensor!(g; dims=score_dims, dtype=Float32, virtual=true, name="MaskedScore")
    return SDPAMaskSubgraph(input, fill, output)
end

operation_graph_mode(::SDPAFwdOp) = CUDNN_OPERATIONGRAPH_MODE_UNIFIED_SDPA_FWD
operation_graph_mode(::SDPABwdOp) = CUDNN_OPERATIONGRAPH_MODE_UNIFIED_SDPA_BWD
operation_graph_mode(::ConvFpropOp) = CUDNN_OPERATIONGRAPH_MODE_CONV_FORWARD
operation_graph_mode(::ConvDgradOp) = CUDNN_OPERATIONGRAPH_MODE_CONV_BWD_DATA
operation_graph_mode(::ConvWgradOp) = CUDNN_OPERATIONGRAPH_MODE_CONV_BWD_FILTER
operation_graph_mode(::ResampleFwdOp) = CUDNN_OPERATIONGRAPH_MODE_RESAMPLE_FWD
operation_graph_mode(::ResampleBwdOp) = CUDNN_OPERATIONGRAPH_MODE_RESAMPLE_BWD
operation_graph_mode(op::NormFwdOp) =
    op.phase == CUDNN_NORM_FWD_TRAINING ? CUDNN_OPERATIONGRAPH_MODE_NORM_FWD_TRAIN :
    CUDNN_OPERATIONGRAPH_MODE_NORM_FWD_INFER
operation_graph_mode(::NormBwdOp) = CUDNN_OPERATIONGRAPH_MODE_NORM_BWD

alphabeta_type(dtype::cudnnDataType_t) =
    dtype == CUDNN_DATA_DOUBLE ? CUDNN_TYPE_DOUBLE : CUDNN_TYPE_FLOAT

pointwise_mode(mode::cudnnPointwiseMode_t) = mode
pointwise_mode(mode::Symbol) =
    mode === :add ? CUDNN_POINTWISE_ADD :
    mode === :mul ? CUDNN_POINTWISE_MUL :
    mode === :sub ? CUDNN_POINTWISE_SUB :
    mode === :div ? CUDNN_POINTWISE_DIV :
    mode === :max ? CUDNN_POINTWISE_MAX :
    mode === :min ? CUDNN_POINTWISE_MIN :
    mode === :sqrt ? CUDNN_POINTWISE_SQRT :
    mode === :rsqrt ? CUDNN_POINTWISE_RSQRT :
    mode === :exp ? CUDNN_POINTWISE_EXP :
    mode === :log ? CUDNN_POINTWISE_LOG :
    mode === :neg ? CUDNN_POINTWISE_NEG :
    mode === :identity ? CUDNN_POINTWISE_IDENTITY :
    mode === :relu ? CUDNN_POINTWISE_RELU_FWD :
    mode === :tanh ? CUDNN_POINTWISE_TANH_FWD :
    mode === :sigmoid ? CUDNN_POINTWISE_SIGMOID_FWD :
    mode === :elu ? CUDNN_POINTWISE_ELU_FWD :
    throw(ArgumentError("unknown cuDNN pointwise mode $mode"))

reduction_mode(mode::cudnnReduceTensorOp_t) = mode
reduction_mode(mode::Symbol) =
    mode === :sum ? CUDNN_REDUCE_TENSOR_ADD :
    mode === :add ? CUDNN_REDUCE_TENSOR_ADD :
    mode === :mul ? CUDNN_REDUCE_TENSOR_MUL :
    mode === :min ? CUDNN_REDUCE_TENSOR_MIN :
    mode === :max ? CUDNN_REDUCE_TENSOR_MAX :
    mode === :avg ? CUDNN_REDUCE_TENSOR_AVG :
    throw(ArgumentError("unknown cuDNN reduction mode $mode"))

graph_conv_mode(mode::cudnnConvolutionMode_t) = mode
graph_conv_mode(mode::Symbol) =
    mode === :cross_correlation ? CUDNN_CROSS_CORRELATION :
    mode === :convolution ? CUDNN_CONVOLUTION :
    throw(ArgumentError("unknown cuDNN convolution mode $mode"))

resample_mode(mode::cudnnResampleMode_t) = mode
resample_mode(mode::Symbol) =
    mode === :maxpool ? CUDNN_RESAMPLE_MAXPOOL :
    mode === :avgpool ? CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING :
    mode === :avgpool_include_padding ? CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING :
    mode === :avgpool_exclude_padding ? CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING :
    mode === :nearest ? CUDNN_RESAMPLE_NEAREST :
    mode === :bilinear ? CUDNN_RESAMPLE_BILINEAR :
    throw(ArgumentError("unknown cuDNN resample mode $mode"))

graph_padding_mode(mode::cudnnPaddingMode_t) = mode
graph_padding_mode(mode::Symbol) =
    mode === :zero ? CUDNN_ZERO_PAD :
    mode === :neg_inf ? CUDNN_NEG_INF_PAD :
    mode === :edge ? CUDNN_EDGE_VAL_PAD :
    throw(ArgumentError("unknown cuDNN padding mode $mode"))

norm_mode(mode::cudnnBackendNormMode_t) = mode
norm_mode(mode::Symbol) =
    mode === :batchnorm ? CUDNN_BATCH_NORM :
    mode === :layernorm ? CUDNN_LAYER_NORM :
    mode === :instancenorm ? CUDNN_INSTANCE_NORM :
    mode === :groupnorm ? CUDNN_GROUP_NORM :
    mode === :rmsnorm ? CUDNN_RMS_NORM :
    throw(ArgumentError("unknown cuDNN norm mode $mode"))

norm_phase(phase::cudnnBackendNormFwdPhase_t) = phase
norm_phase(phase::Symbol) =
    phase === :training ? CUDNN_NORM_FWD_TRAINING :
    phase === :inference ? CUDNN_NORM_FWD_INFERENCE :
    throw(ArgumentError("unknown cuDNN norm phase $phase"))

function broadcast_dims(ts::Tensor...)
    isempty(ts) && throw(ArgumentError("at least one tensor is required"))
    rank = length(first(ts).dims)
    all(t -> length(t.dims) == rank, ts) ||
        throw(DimensionMismatch("pointwise tensors must have the same rank"))
    dims = Vector{Int64}(undef, rank)
    for i in 1:rank
        d = maximum(t.dims[i] for t in ts)
        all(t -> t.dims[i] == d || t.dims[i] == 1, ts) ||
            throw(DimensionMismatch("pointwise tensor dimensions are not broadcastable"))
        dims[i] = d
    end
    return dims
end

function broadcast_dim_vectors(vs::AbstractVector...)
    isempty(vs) && return Int64[]
    n = maximum(length, vs)
    dims = Int64[]
    for i in 1:n
        vals = map(vs) do v
            j = length(v) - n + i
            j < 1 ? Int64(1) : Int64(v[j])
        end
        d = maximum(vals)
        all(x -> x == d || x == 1, vals) ||
            throw(DimensionMismatch("batch dimensions are not broadcastable"))
        push!(dims, d)
    end
    return dims
end

function spatial_vector(v, rank; default=nothing)
    v === nothing && (v = default)
    v isa Integer && return fill(Int64(v), rank)
    out = collect(Int64, v)
    length(out) == rank ||
        throw(DimensionMismatch("expected $rank spatial values, got $(length(out))"))
    return out
end

function require_positive(name, xs)
    all(>(0), xs) || throw(ArgumentError("$name must be positive"))
    return xs
end

function norm_channel_dims(x::Tensor)
    length(x.dims) >= 3 || throw(ArgumentError("norm input must have rank >= 3"))
    dims = fill(Int64(1), length(x.dims))
    dims[end-1] = x.dims[end-1]
    return dims
end

function check_norm_param(name, t::Tensor, dims)
    t.dims == dims || throw(DimensionMismatch("$name dimensions must be $(Tuple(dims))"))
    return t
end

norm_scalar_type(g::Graph) = g.compute_dtype == CUDNN_DATA_DOUBLE ? Float64 : Float32

norm_stat_dtype(dtype) = dtype == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT

function check_norm_dtype(name, t::Tensor, dtype)
    t.dtype === nothing && (t.dtype = dtype)
    t.dtype == dtype ||
        throw(ArgumentError("$name must have $(juliaDataType(dtype)) dtype"))
    return t
end

function pointwise!(g::Graph, mode, x::Tensor, inputs::Tensor...;
                    y::Union{Nothing,Tensor}=nothing, compute_dtype=g.compute_dtype,
                    nan_propagation::cudnnNanPropagation_t=CUDNN_NOT_PROPAGATE_NAN,
                    relu_lower_clip::Real=0, relu_upper_clip::Real=Inf,
                    relu_lower_clip_slope::Real=0, elu_alpha::Real=1,
                    softplus_beta::Real=1, swish_beta::Real=1,
                    axis::Union{Nothing,Integer}=nothing,
                    alpha1::Real=1, alpha2::Real=0, name::String="Y")
    length(inputs) <= 2 || throw(ArgumentError("pointwise! accepts at most three inputs"))
    b = length(inputs) >= 1 ? inputs[1] : nothing
    t = length(inputs) == 2 ? inputs[2] : nothing
    tensors = (x, inputs...)
    outdims = broadcast_dims(tensors...)
    if y === nothing
        y = tensor!(g; dims=outdims, dtype=nothing, virtual=true, name,
                    backend_order=x.backend_order)
    else
        y.dims == outdims ||
            throw(DimensionMismatch("pointwise output dimensions must match broadcast dimensions"))
    end
    push!(g.ops, PointwiseOp(pointwise_mode(mode), x, y, b, t, graph_dtype(compute_dtype),
                             nan_propagation, Float64(relu_lower_clip),
                             Float64(relu_upper_clip), Float64(relu_lower_clip_slope),
                             Float64(elu_alpha), Float64(softplus_beta),
                             Float64(swish_beta),
                             axis === nothing ? nothing : Int64(axis),
                             Float32(alpha1), Float32(alpha2)))
    return y
end

function conv_output_dim(x, w, pre, post, stride, dilation)
    return fld(x + pre + post - dilation * (w - 1) - 1, stride) + 1
end

function conv_output_dims(x_spatial, w_spatial, pre, post, stride, dilation)
    [conv_output_dim(x_spatial[i], w_spatial[i], pre[i], post[i], stride[i], dilation[i])
     for i in eachindex(x_spatial)]
end

function convolution_group_count(input_channels, filter_input_channels, output_channels)
    filter_input_channels > 0 || throw(ArgumentError("filter channels must be positive"))
    input_channels % filter_input_channels == 0 ||
        throw(DimensionMismatch("input channels must be a multiple of filter channels"))
    groups = input_channels ÷ filter_input_channels
    output_channels % groups == 0 ||
        throw(DimensionMismatch("output channels must be divisible by groups"))
    return groups
end

function conv_fprop!(g::Graph, x::Tensor, w::Tensor;
                     y::Union{Nothing,Tensor}=nothing,
                     pre_padding=0, post_padding=nothing,
                     stride=1, dilation=1, mode=:cross_correlation,
                     compute_dtype=g.compute_dtype, alpha::Real=1, beta::Real=0,
                     name::String="Y")
    length(x.dims) >= 3 || throw(ArgumentError("convolution input must have rank >= 3"))
    length(w.dims) == length(x.dims) ||
        throw(DimensionMismatch("convolution filter rank must match input rank"))
    rank = length(x.dims) - 2
    pre = spatial_vector(pre_padding, rank)
    post = spatial_vector(post_padding, rank; default=pre_padding)
    str = spatial_vector(stride, rank)
    dil = spatial_vector(dilation, rank)
    all(>(0), str) || throw(ArgumentError("convolution stride must be positive"))
    all(>(0), dil) || throw(ArgumentError("convolution dilation must be positive"))

    cin = x.dims[end-1]
    cfilter = w.dims[end-1]
    convolution_group_count(cin, cfilter, w.dims[end])
    out_spatial = conv_output_dims(x.dims[1:rank], w.dims[1:rank], pre, post, str, dil)
    all(>(0), out_spatial) ||
        throw(DimensionMismatch("convolution output spatial dimensions must be positive"))
    outdims = Int64[out_spatial; w.dims[end]; x.dims[end]]
    if y === nothing
        y = tensor!(g; dims=outdims, dtype=x.dtype, virtual=true, name,
                    backend_order=x.backend_order)
    else
        y.dims == outdims ||
            throw(DimensionMismatch("convolution output dimensions must be $(Tuple(outdims))"))
    end
    push!(g.ops, ConvFpropOp(x, w, y, pre, post, str, dil, graph_conv_mode(mode),
                             graph_dtype(compute_dtype), Float64(alpha), Float64(beta)))
    return y
end

resample_output_dim(x, window, pre, post, stride) =
    fld(x + pre + post - window, stride) + 1

function resample_output_dims(x_spatial, window, pre, post, stride)
    [resample_output_dim(x_spatial[i], window[i], pre[i], post[i], stride[i])
     for i in eachindex(x_spatial)]
end

function resample_fwd!(g::Graph, x::Tensor; y::Union{Nothing,Tensor}=nothing,
                       index::Union{Nothing,Tensor}=nothing, mode, window,
                       stride=window, pre_padding=0, post_padding=nothing,
                       padding_mode=:zero,
                       nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                       compute_dtype=g.compute_dtype, alpha::Real=1, beta::Real=0,
                       generate_index::Bool=false, name::String="Y",
                       index_name::String="Index")
    length(x.dims) >= 3 || throw(ArgumentError("resample input must have rank >= 3"))
    rank = length(x.dims) - 2
    win = require_positive("resample window", spatial_vector(window, rank))
    str = require_positive("resample stride", spatial_vector(stride, rank))
    pre = spatial_vector(pre_padding, rank)
    post = spatial_vector(post_padding, rank; default=pre_padding)
    rmode = resample_mode(mode)
    pmode = graph_padding_mode(padding_mode)
    out_spatial = resample_output_dims(x.dims[1:rank], win, pre, post, str)
    all(>(0), out_spatial) ||
        throw(DimensionMismatch("resample output spatial dimensions must be positive"))
    outdims = Int64[out_spatial; x.dims[end-1]; x.dims[end]]

    if y === nothing
        y = tensor!(g; dims=outdims, dtype=x.dtype, virtual=true, name,
                    backend_order=x.backend_order)
    else
        y.dims == outdims ||
            throw(DimensionMismatch("resample output dimensions must be $(Tuple(outdims))"))
    end

    if generate_index && index === nothing
        rmode == CUDNN_RESAMPLE_MAXPOOL ||
            throw(ArgumentError("resample indices are only valid for maxpool"))
        index = tensor!(g; dims=outdims, dtype=Int8, virtual=true, name=index_name,
                        backend_order=x.backend_order)
    end
    if index !== nothing
        rmode == CUDNN_RESAMPLE_MAXPOOL ||
            throw(ArgumentError("resample indices are only valid for maxpool"))
        index.dims == outdims ||
            throw(DimensionMismatch("resample index dimensions must match output"))
    end

    push!(g.ops, ResampleFwdOp(x, y, index, rmode, pmode, win, pre, post, str,
                               graph_dtype(compute_dtype), nan_propagation,
                               Float64(alpha), Float64(beta)))
    return index === nothing ? y : (y, index)
end

function resample_bwd!(g::Graph, dy::Tensor; dx::Union{Nothing,Tensor}=nothing,
                       x::Union{Nothing,Tensor}=nothing,
                       y::Union{Nothing,Tensor}=nothing,
                       index::Union{Nothing,Tensor}=nothing, x_dims=nothing,
                       mode, window, stride=window, pre_padding=0,
                       post_padding=nothing, padding_mode=:zero,
                       nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                       compute_dtype=g.compute_dtype, alpha::Real=1, beta::Real=0,
                       name::String="dX")
    length(dy.dims) >= 3 || throw(ArgumentError("resample dY must have rank >= 3"))
    rank = length(dy.dims) - 2
    win = require_positive("resample window", spatial_vector(window, rank))
    str = require_positive("resample stride", spatial_vector(stride, rank))
    pre = spatial_vector(pre_padding, rank)
    post = spatial_vector(post_padding, rank; default=pre_padding)
    rmode = resample_mode(mode)
    pmode = graph_padding_mode(padding_mode)

    xdims = dx === nothing ? (x === nothing ? x_dims : x.dims) : dx.dims
    xdims === nothing && throw(ArgumentError("resample_bwd! requires dx, x, or x_dims"))
    xdims = collect(Int64, xdims)
    length(xdims) == length(dy.dims) ||
        throw(DimensionMismatch("x dimensions must match dY rank"))
    xdims[end-1:end] == dy.dims[end-1:end] ||
        throw(DimensionMismatch("x and dY channel/batch dimensions must match"))
    expected = resample_output_dims(xdims[1:rank], win, pre, post, str)
    expected == dy.dims[1:rank] ||
        throw(DimensionMismatch("dY spatial dimensions must be $(Tuple(expected))"))

    if dx === nothing
        dx = tensor!(g; dims=xdims, dtype=dy.dtype, virtual=true, name,
                     backend_order=dy.backend_order)
    else
        dx.dims == xdims || throw(DimensionMismatch("dx dimensions do not match x_dims"))
    end
    x === nothing || x.dims == xdims ||
        throw(DimensionMismatch("x dimensions must match dx"))
    y === nothing || y.dims == dy.dims ||
        throw(DimensionMismatch("y dimensions must match dY"))
    if index !== nothing
        rmode == CUDNN_RESAMPLE_MAXPOOL ||
            throw(ArgumentError("resample indices are only valid for maxpool"))
        index.dims == dy.dims ||
            throw(DimensionMismatch("resample index dimensions must match dY"))
    end

    push!(g.ops, ResampleBwdOp(dx, dy, x, y, index, rmode, pmode, win, pre, post, str,
                               graph_dtype(compute_dtype), nan_propagation,
                               Float64(alpha), Float64(beta)))
    return dx
end

function conv_dgrad!(g::Graph, dy::Tensor, w::Tensor;
                     dx::Union{Nothing,Tensor}=nothing, x_dims=nothing,
                     pre_padding=0, post_padding=nothing,
                     stride=1, dilation=1, mode=:cross_correlation,
                     compute_dtype=g.compute_dtype, alpha::Real=1, beta::Real=0,
                     name::String="dX")
    length(dy.dims) >= 3 || throw(ArgumentError("convolution dY must have rank >= 3"))
    length(w.dims) == length(dy.dims) ||
        throw(DimensionMismatch("convolution filter rank must match dY rank"))
    rank = length(dy.dims) - 2
    pre = spatial_vector(pre_padding, rank)
    post = spatial_vector(post_padding, rank; default=pre_padding)
    str = spatial_vector(stride, rank)
    dil = spatial_vector(dilation, rank)
    all(>(0), str) || throw(ArgumentError("convolution stride must be positive"))
    all(>(0), dil) || throw(ArgumentError("convolution dilation must be positive"))

    xdims = dx === nothing ? x_dims : dx.dims
    xdims === nothing &&
        throw(ArgumentError("conv_dgrad! requires dx or x_dims"))
    xdims = collect(Int64, xdims)
    length(xdims) == length(dy.dims) ||
        throw(DimensionMismatch("x dimensions must match dY rank"))
    xdims[end] == dy.dims[end] || throw(DimensionMismatch("x and dY batch sizes must match"))
    convolution_group_count(xdims[end-1], w.dims[end-1], w.dims[end])
    dy.dims[end-1] == w.dims[end] ||
        throw(DimensionMismatch("dY channels must match filter output channels"))
    expected = conv_output_dims(xdims[1:rank], w.dims[1:rank], pre, post, str, dil)
    expected == dy.dims[1:rank] ||
        throw(DimensionMismatch("dY spatial dimensions must be $(Tuple(expected))"))

    if dx === nothing
        dx = tensor!(g; dims=xdims, dtype=dy.dtype, virtual=true, name,
                     backend_order=dy.backend_order)
    else
        dx.dims == xdims || throw(DimensionMismatch("dx dimensions do not match x_dims"))
    end
    push!(g.ops, ConvDgradOp(dy, w, dx, pre, post, str, dil, graph_conv_mode(mode),
                             graph_dtype(compute_dtype), Float64(alpha), Float64(beta)))
    return dx
end

function conv_wgrad!(g::Graph, dy::Tensor, x::Tensor;
                     dw::Union{Nothing,Tensor}=nothing, w_dims=nothing,
                     pre_padding=0, post_padding=nothing,
                     stride=1, dilation=1, mode=:cross_correlation,
                     compute_dtype=g.compute_dtype, alpha::Real=1, beta::Real=0,
                     name::String="dW")
    length(x.dims) >= 3 || throw(ArgumentError("convolution input must have rank >= 3"))
    length(dy.dims) == length(x.dims) ||
        throw(DimensionMismatch("convolution dY rank must match input rank"))
    rank = length(x.dims) - 2
    pre = spatial_vector(pre_padding, rank)
    post = spatial_vector(post_padding, rank; default=pre_padding)
    str = spatial_vector(stride, rank)
    dil = spatial_vector(dilation, rank)
    all(>(0), str) || throw(ArgumentError("convolution stride must be positive"))
    all(>(0), dil) || throw(ArgumentError("convolution dilation must be positive"))

    wdims = dw === nothing ? w_dims : dw.dims
    wdims === nothing &&
        throw(ArgumentError("conv_wgrad! requires dw or w_dims"))
    wdims = collect(Int64, wdims)
    length(wdims) == length(x.dims) ||
        throw(DimensionMismatch("filter dimensions must match input rank"))
    x.dims[end] == dy.dims[end] || throw(DimensionMismatch("input and dY batch sizes must match"))
    convolution_group_count(x.dims[end-1], wdims[end-1], wdims[end])
    dy.dims[end-1] == wdims[end] ||
        throw(DimensionMismatch("dY channels must match filter output channels"))
    expected = conv_output_dims(x.dims[1:rank], wdims[1:rank], pre, post, str, dil)
    expected == dy.dims[1:rank] ||
        throw(DimensionMismatch("dY spatial dimensions must be $(Tuple(expected))"))

    if dw === nothing
        dw = tensor!(g; dims=wdims, dtype=x.dtype, virtual=true, name,
                     backend_order=x.backend_order)
    else
        dw.dims == wdims || throw(DimensionMismatch("dw dimensions do not match w_dims"))
    end
    push!(g.ops, ConvWgradOp(x, dy, dw, pre, post, str, dil, graph_conv_mode(mode),
                             graph_dtype(compute_dtype), Float64(alpha), Float64(beta)))
    return dw
end

function norm_fwd!(g::Graph, x::Tensor, scale::Tensor, bias::Tensor;
                   y::Union{Nothing,Tensor}=nothing,
                   mean::Union{Nothing,Tensor}=nothing,
                   inv_variance::Union{Nothing,Tensor}=nothing,
                   mode=:batchnorm, phase=:training,
                   epsilon::Union{Nothing,Tensor}=nothing,
                   momentum::Union{Nothing,Tensor}=nothing,
                   input_running_mean::Union{Nothing,Tensor}=nothing,
                   input_running_var::Union{Nothing,Tensor}=nothing,
                   output_running_mean::Union{Nothing,Tensor}=nothing,
                   output_running_var::Union{Nothing,Tensor}=nothing,
                   name::String="Y")
    pdims = norm_channel_dims(x)
    check_norm_param("norm scale", scale, pdims)
    check_norm_param("norm bias", bias, pdims)
    xdtype = something(x.dtype, g.io_dtype)
    check_norm_dtype("norm input", x, xdtype)
    stat_dtype = norm_stat_dtype(xdtype)
    check_norm_dtype("norm scale", scale, stat_dtype)
    check_norm_dtype("norm bias", bias, stat_dtype)
    nphase = norm_phase(phase)
    nmode = norm_mode(mode)

    if y === nothing
        y = tensor!(g; dims=x.dims, dtype=xdtype, virtual=true, name,
                    backend_order=x.backend_order)
    else
        y.dims == x.dims || throw(DimensionMismatch("norm output dimensions must match input"))
        check_norm_dtype("norm output", y, xdtype)
    end

    if nphase == CUDNN_NORM_FWD_TRAINING
        mean === nothing &&
            (mean = tensor!(g; dims=pdims, dtype=stat_dtype, virtual=true, name="Mean",
                            backend_order=x.backend_order))
        inv_variance === nothing &&
            (inv_variance = tensor!(g; dims=pdims, dtype=stat_dtype, virtual=true,
                                    name="InvVariance", backend_order=x.backend_order))
        check_norm_param("norm mean", mean, pdims)
        check_norm_param("norm inv_variance", inv_variance, pdims)
        check_norm_dtype("norm mean", mean, stat_dtype)
        check_norm_dtype("norm inv_variance", inv_variance, stat_dtype)
        epsilon === nothing &&
            (epsilon = scalar!(g, norm_scalar_type(g); rank=length(x.dims), name="Epsilon"))
        check_norm_dtype("norm epsilon", epsilon, graph_dtype(norm_scalar_type(g)))
        has_running = input_running_mean !== nothing || input_running_var !== nothing ||
                      output_running_mean !== nothing || output_running_var !== nothing
        if has_running
            if input_running_mean === nothing || input_running_var === nothing ||
               output_running_mean === nothing || output_running_var === nothing ||
               momentum === nothing
                throw(ArgumentError("running stats require input/output mean/var and momentum"))
            end
            check_norm_param("input_running_mean", input_running_mean, pdims)
            check_norm_param("input_running_var", input_running_var, pdims)
            check_norm_param("output_running_mean", output_running_mean, pdims)
            check_norm_param("output_running_var", output_running_var, pdims)
            check_norm_dtype("input_running_mean", input_running_mean, stat_dtype)
            check_norm_dtype("input_running_var", input_running_var, stat_dtype)
            check_norm_dtype("output_running_mean", output_running_mean, stat_dtype)
            check_norm_dtype("output_running_var", output_running_var, stat_dtype)
            check_norm_dtype("norm momentum", momentum, graph_dtype(norm_scalar_type(g)))
        end
    else
        mean === nothing && throw(ArgumentError("norm inference requires mean"))
        inv_variance === nothing &&
            throw(ArgumentError("norm inference requires inv_variance"))
        check_norm_param("norm mean", mean, pdims)
        check_norm_param("norm inv_variance", inv_variance, pdims)
        check_norm_dtype("norm mean", mean, stat_dtype)
        check_norm_dtype("norm inv_variance", inv_variance, stat_dtype)
        epsilon = nothing
        momentum = nothing
    end

    push!(g.ops, NormFwdOp(x, mean, inv_variance, scale, bias, epsilon, momentum,
                           input_running_mean, input_running_var, output_running_mean,
                           output_running_var, y, nmode, nphase))
    return nphase == CUDNN_NORM_FWD_TRAINING ?
           (y, mean, inv_variance, output_running_mean, output_running_var) : y
end

function norm_bwd!(g::Graph, dy::Tensor, x::Tensor, scale::Tensor, mean::Tensor,
                   inv_variance::Tensor; dx::Union{Nothing,Tensor}=nothing,
                   dscale::Union{Nothing,Tensor}=nothing,
                   dbias::Union{Nothing,Tensor}=nothing, mode=:batchnorm,
                   name::String="dX")
    dy.dims == x.dims || throw(DimensionMismatch("norm dY dimensions must match input"))
    xdtype = something(x.dtype, g.io_dtype)
    check_norm_dtype("norm input", x, xdtype)
    stat_dtype = norm_stat_dtype(xdtype)
    check_norm_dtype("norm dY", dy, xdtype)
    pdims = norm_channel_dims(x)
    check_norm_param("norm scale", scale, pdims)
    check_norm_param("norm mean", mean, pdims)
    check_norm_param("norm inv_variance", inv_variance, pdims)
    check_norm_dtype("norm scale", scale, stat_dtype)
    check_norm_dtype("norm mean", mean, stat_dtype)
    check_norm_dtype("norm inv_variance", inv_variance, stat_dtype)
    dx === nothing && (dx = tensor!(g; dims=x.dims, dtype=x.dtype, virtual=true, name,
                                    backend_order=x.backend_order))
    dx.dims == x.dims || throw(DimensionMismatch("norm dX dimensions must match input"))
    check_norm_dtype("norm dX", dx, xdtype)
    dscale === nothing &&
        (dscale = tensor!(g; dims=pdims, dtype=scale.dtype, virtual=true,
                          name="dScale", backend_order=x.backend_order))
    dbias === nothing &&
        (dbias = tensor!(g; dims=pdims, dtype=scale.dtype, virtual=true,
                         name="dBias", backend_order=x.backend_order))
    check_norm_param("norm dscale", dscale, pdims)
    check_norm_param("norm dbias", dbias, pdims)
    check_norm_dtype("norm dscale", dscale, stat_dtype)
    check_norm_dtype("norm dbias", dbias, stat_dtype)
    push!(g.ops, NormBwdOp(dy, x, scale, mean, inv_variance, dx, dscale, dbias,
                           norm_mode(mode)))
    return dx, dscale, dbias
end

function matmul!(g::Graph, a::Tensor, b::Tensor;
                 c::Union{Nothing,Tensor}=nothing, compute_dtype=g.compute_dtype,
                 name::String="C")
    length(a.dims) >= 3 && length(b.dims) >= 3 ||
        throw(ArgumentError("matmul! tensors must have rank >= 3"))
    a.dims[2] == b.dims[1] ||
        throw(DimensionMismatch("matmul! inner dimensions must match"))
    batch = broadcast_dim_vectors(a.dims[3:end], b.dims[3:end])
    cdims = Int64[a.dims[1]; b.dims[2]; batch...]
    if c === nothing
        c = tensor!(g; dims=cdims, dtype=nothing, virtual=true, name)
    else
        c.dims == cdims ||
            throw(DimensionMismatch("matmul output dimensions must be $(Tuple(cdims))"))
    end
    push!(g.ops, MatmulOp(a, b, c, graph_dtype(compute_dtype)))
    return c
end

function reduction!(g::Graph, mode, x::Tensor;
                    y::Union{Nothing,Tensor}=nothing, dims,
                    compute_dtype=g.compute_dtype, deterministic::Bool=false,
                    name::String="Y")
    rdims = Set(Int.(dims isa Integer ? (dims,) : Tuple(dims)))
    all(d -> 1 <= d <= length(x.dims), rdims) ||
        throw(ArgumentError("reduction dims out of range"))
    outdims = [i in rdims ? Int64(1) : d for (i, d) in enumerate(x.dims)]
    y === nothing && (y = tensor!(g; dims=outdims, dtype=nothing, virtual=true, name,
                                  backend_order=x.backend_order))
    length(y.dims) == length(x.dims) ||
        throw(DimensionMismatch("reduction output rank must match input rank"))
    for i in eachindex(x.dims)
        expected = i in rdims ? 1 : x.dims[i]
        y.dims[i] == expected ||
            throw(DimensionMismatch("reduction output dimensions do not match reduction dims"))
    end
    push!(g.ops, ReductionOp(reduction_mode(mode), x, y, graph_dtype(compute_dtype),
                             deterministic))
    return y
end

function sdpa_fwd!(g::Graph, q::Tensor, k::Tensor, v::Tensor;
                   o::Union{Nothing,Tensor}=nothing,
                   scale::Union{Nothing,Tensor}=nothing,
                   stats::Union{Nothing,Tensor,Bool}=nothing,
                   seq_len_q::Union{Nothing,Tensor}=nothing,
                   seq_len_kv::Union{Nothing,Tensor}=nothing,
                   causal::Bool=false, dropout_p::Real=0,
                   bias=nothing)
    dropout_p == 0 || throw(ArgumentError("cuDNN SDPA dropout is not implemented yet"))
    bias === nothing || throw(ArgumentError("cuDNN SDPA bias is not implemented yet"))

    length(q.dims) == 4 || throw(ArgumentError("q must be rank 4"))
    length(k.dims) == 4 || throw(ArgumentError("k must be rank 4"))
    length(v.dims) == 4 || throw(ArgumentError("v must be rank 4"))
    d, hq, sq, b = q.dims
    dk, hk, skv, bk = k.dims
    dv, hv, skvv, bv = v.dims
    dk == d || throw(DimensionMismatch("q and k head dimensions must match"))
    dv == d || throw(DimensionMismatch("q and v head dimensions must match"))
    hk == hv || throw(DimensionMismatch("k and v head counts must match"))
    skv == skvv || throw(DimensionMismatch("k and v sequence lengths must match"))
    bk == b && bv == b || throw(DimensionMismatch("q, k, and v batch sizes must match"))
    hq % hk == 0 || throw(DimensionMismatch("q head count must be a multiple of k/v heads"))
    check_sdpa_sequence_lengths(seq_len_q, seq_len_kv, b)

    o === nothing && (o = tensor!(g; dims=(d, hq, sq, b), dtype=q.dtype, virtual=true,
                                  name="O"))
    scale === nothing && (scale = scalar!(g, Float32; rank=4, name="Scale"))
    stats === true && (stats = tensor!(g; dims=(1, hq, sq, b), dtype=Float32,
                                       name="Stats", output=true))
    stats === false && (stats = nothing)

    foreach(sdpa_tensor!, (q, k, v, o, scale))
    stats === nothing || sdpa_tensor!(stats)
    seq_len_q === nothing || foreach(sdpa_tensor!, (seq_len_q, seq_len_kv))

    mask_subgraph = causal ? sdpa_causal_subgraph!(g, skv, sq, hq, b) : nothing

    push!(g.ops, SDPAFwdOp(q, k, v, o, scale, stats, seq_len_q, seq_len_kv,
                           mask_subgraph))
    return stats === nothing ? o : (o, stats)
end

function sdpa_bwd!(g::Graph, q::Tensor, k::Tensor, v::Tensor, o::Tensor, dO::Tensor,
                   stats::Tensor;
                   dQ::Union{Nothing,Tensor}=nothing,
                   dK::Union{Nothing,Tensor}=nothing,
                   dV::Union{Nothing,Tensor}=nothing,
                   scale::Union{Nothing,Tensor}=nothing,
                   seq_len_q::Union{Nothing,Tensor}=nothing,
                   seq_len_kv::Union{Nothing,Tensor}=nothing,
                   causal::Bool=false, dropout_p::Real=0, bias=nothing)
    dropout_p == 0 || throw(ArgumentError("cuDNN SDPA backward dropout is not implemented yet"))
    bias === nothing || throw(ArgumentError("cuDNN SDPA backward bias is not implemented yet"))

    length(q.dims) == 4 || throw(ArgumentError("q must be rank 4"))
    length(k.dims) == 4 || throw(ArgumentError("k must be rank 4"))
    length(v.dims) == 4 || throw(ArgumentError("v must be rank 4"))
    d, hq, sq, b = q.dims
    dk, hk, skv, bk = k.dims
    dv, hv, skvv, bv = v.dims
    dk == d || throw(DimensionMismatch("q and k head dimensions must match"))
    dv == d || throw(DimensionMismatch("q and v head dimensions must match"))
    hk == hv || throw(DimensionMismatch("k and v head counts must match"))
    skv == skvv || throw(DimensionMismatch("k and v sequence lengths must match"))
    bk == b && bv == b || throw(DimensionMismatch("q, k, and v batch sizes must match"))
    hq % hk == 0 || throw(DimensionMismatch("q head count must be a multiple of k/v heads"))
    o.dims == q.dims || throw(DimensionMismatch("o dimensions must match q"))
    dO.dims == q.dims || throw(DimensionMismatch("dO dimensions must match q"))
    stats.dims == [1, hq, sq, b] ||
        throw(DimensionMismatch("stats dimensions must be $((1, hq, sq, b))"))
    (stats.dtype === nothing || stats.dtype == CUDNN_DATA_FLOAT) ||
        throw(ArgumentError("stats must have Float32 dtype"))
    check_sdpa_sequence_lengths(seq_len_q, seq_len_kv, b)

    dQ === nothing && (dQ = tensor!(g; dims=q.dims, dtype=q.dtype, virtual=true,
                                    name="dQ"))
    dK === nothing && (dK = tensor!(g; dims=k.dims, dtype=k.dtype, virtual=true,
                                    name="dK"))
    dV === nothing && (dV = tensor!(g; dims=v.dims, dtype=v.dtype, virtual=true,
                                    name="dV"))
    dQ.dims == q.dims || throw(DimensionMismatch("dQ dimensions must match q"))
    dK.dims == k.dims || throw(DimensionMismatch("dK dimensions must match k"))
    dV.dims == v.dims || throw(DimensionMismatch("dV dimensions must match v"))
    scale === nothing && (scale = scalar!(g, Float32; rank=4, name="Scale"))

    foreach(sdpa_tensor!, (q, k, v, o, dO, stats, scale, dQ, dK, dV))
    seq_len_q === nothing || foreach(sdpa_tensor!, (seq_len_q, seq_len_kv))

    mask_subgraph = causal ? sdpa_causal_subgraph!(g, skv, sq, hq, b) : nothing

    push!(g.ops, SDPABwdOp(q, k, v, o, dO, stats, scale, dQ, dK, dV,
                           seq_len_q, seq_len_kv, mask_subgraph))
    return dQ, dK, dV
end

function lower(op::PointwiseOp, ctx::LoweringContext)
    pwdesc = track!(ctx, pointwise_descriptor(mode=op.mode, compute_type=op.compute_dtype,
                                             nan_propagation=op.nan_propagation,
                                             relu_lower_clip=op.relu_lower_clip,
                                             relu_upper_clip=op.relu_upper_clip,
                                             relu_lower_clip_slope=op.relu_lower_clip_slope,
                                             elu_alpha=op.elu_alpha,
                                             softplus_beta=op.softplus_beta,
                                             swish_beta=op.swish_beta,
                                             axis=op.axis))
    pointwise_operation(pwdesc, desc(ctx, op.x), desc(ctx, op.y);
                        b=op.b === nothing ? nothing : desc(ctx, op.b),
                        t=op.t === nothing ? nothing : desc(ctx, op.t),
                        alpha1=op.alpha1, alpha2=op.alpha2)
end

function lower(op::MatmulOp, ctx::LoweringContext)
    matdesc = track!(ctx, matmul_descriptor(compute_type=op.compute_dtype))
    matmul_operation(matdesc, desc(ctx, op.b), desc(ctx, op.a), desc(ctx, op.c))
end

function lower(op::ReductionOp, ctx::LoweringContext)
    reddesc = track!(ctx, reduction_descriptor(mode=op.mode, compute_type=op.compute_dtype,
                                               deterministic=op.deterministic))
    reduction_operation(reddesc, desc(ctx, op.x), desc(ctx, op.y))
end

function lower(op::ConvFpropOp, ctx::LoweringContext)
    convdesc = track!(ctx, backend_convolution_descriptor(compute_type=op.compute_dtype,
                                                          mode=op.mode,
                                                          pre_padding=reverse(op.pre_padding),
                                                          post_padding=reverse(op.post_padding),
                                                          dilation=reverse(op.dilation),
                                                          stride=reverse(op.stride)))
    convolution_forward_operation(convdesc, desc(ctx, op.x), desc(ctx, op.w),
                                  desc(ctx, op.y); alpha=op.alpha, beta=op.beta,
                                  alphabeta_type=alphabeta_type(op.compute_dtype))
end

function lower(op::ConvDgradOp, ctx::LoweringContext)
    convdesc = track!(ctx, backend_convolution_descriptor(compute_type=op.compute_dtype,
                                                          mode=op.mode,
                                                          pre_padding=reverse(op.pre_padding),
                                                          post_padding=reverse(op.post_padding),
                                                          dilation=reverse(op.dilation),
                                                          stride=reverse(op.stride)))
    convolution_data_backward_operation(convdesc, desc(ctx, op.w), desc(ctx, op.dy),
                                        desc(ctx, op.dx); alpha=op.alpha, beta=op.beta,
                                        alphabeta_type=alphabeta_type(op.compute_dtype))
end

function lower(op::ConvWgradOp, ctx::LoweringContext)
    convdesc = track!(ctx, backend_convolution_descriptor(compute_type=op.compute_dtype,
                                                          mode=op.mode,
                                                          pre_padding=reverse(op.pre_padding),
                                                          post_padding=reverse(op.post_padding),
                                                          dilation=reverse(op.dilation),
                                                          stride=reverse(op.stride)))
    convolution_filter_backward_operation(convdesc, desc(ctx, op.x), desc(ctx, op.dy),
                                          desc(ctx, op.dw); alpha=op.alpha, beta=op.beta,
                                          alphabeta_type=alphabeta_type(op.compute_dtype))
end

function lower(op::ResampleFwdOp, ctx::LoweringContext)
    rsdesc = track!(ctx, backend_resample_descriptor(mode=op.mode,
                                                     compute_type=op.compute_dtype,
                                                     window=reverse(op.window),
                                                     pre_padding=reverse(op.pre_padding),
                                                     post_padding=reverse(op.post_padding),
                                                     stride=reverse(op.stride),
                                                     nan_propagation=op.nan_propagation,
                                                     padding_mode=op.padding_mode))
    resample_forward_operation(rsdesc, desc(ctx, op.x), desc(ctx, op.y);
                               index=op.index === nothing ? nothing : desc(ctx, op.index),
                               alpha=op.alpha, beta=op.beta)
end

function lower(op::ResampleBwdOp, ctx::LoweringContext)
    rsdesc = track!(ctx, backend_resample_descriptor(mode=op.mode,
                                                     compute_type=op.compute_dtype,
                                                     window=reverse(op.window),
                                                     pre_padding=reverse(op.pre_padding),
                                                     post_padding=reverse(op.post_padding),
                                                     stride=reverse(op.stride),
                                                     nan_propagation=op.nan_propagation,
                                                     padding_mode=op.padding_mode))
    resample_backward_operation(rsdesc, desc(ctx, op.dx), desc(ctx, op.dy);
                                x=op.x === nothing ? nothing : desc(ctx, op.x),
                                y=op.y === nothing ? nothing : desc(ctx, op.y),
                                index=op.index === nothing ? nothing : desc(ctx, op.index),
                                alpha=op.alpha, beta=op.beta)
end

function lower(op::NormFwdOp, ctx::LoweringContext)
    norm_forward_operation(mode=op.mode, phase=op.phase,
                           x=desc(ctx, op.x),
                           mean=op.mean === nothing ? nothing : desc(ctx, op.mean),
                           inv_variance=op.inv_variance === nothing ? nothing :
                                        desc(ctx, op.inv_variance),
                           scale=desc(ctx, op.scale), bias=desc(ctx, op.bias),
                           epsilon=op.epsilon === nothing ? nothing : desc(ctx, op.epsilon),
                           momentum=op.momentum === nothing ? nothing : desc(ctx, op.momentum),
                           input_running_mean=op.input_running_mean === nothing ? nothing :
                                              desc(ctx, op.input_running_mean),
                           input_running_var=op.input_running_var === nothing ? nothing :
                                             desc(ctx, op.input_running_var),
                           output_running_mean=op.output_running_mean === nothing ? nothing :
                                               desc(ctx, op.output_running_mean),
                           output_running_var=op.output_running_var === nothing ? nothing :
                                              desc(ctx, op.output_running_var),
                           y=desc(ctx, op.y))
end

function lower(op::NormBwdOp, ctx::LoweringContext)
    norm_backward_operation(mode=op.mode, x=desc(ctx, op.x), mean=desc(ctx, op.mean),
                            inv_variance=desc(ctx, op.inv_variance),
                            dy=desc(ctx, op.dy), scale=desc(ctx, op.scale),
                            dscale=desc(ctx, op.dscale), dbias=desc(ctx, op.dbias),
                            dx=desc(ctx, op.dx))
end

function lower_sdpa_mask_subgraph!(d, ctx, mask)
    mask_op = track!(ctx, diagonal_band_mask_operation(desc(ctx, mask.input),
                                                       desc(ctx, mask.fill),
                                                       desc(ctx, mask.output);
                                                       comparison_mode=CUDNN_POINTWISE_CMP_GE))
    d[:subgraph] = track!(ctx, operation_graph([mask_op]))
    d[:subgraph_input_uid] = mask.input.uid
    d[:subgraph_output_uid] = mask.output.uid
    return
end

function lower(op::SDPAFwdOp, ctx::LoweringContext)
    make_descriptor(:operation_sdpa_fwd) do d
        d[:qdesc] = desc(ctx, op.q)
        d[:kdesc] = desc(ctx, op.k)
        d[:vdesc] = desc(ctx, op.v)
        d[:odesc] = desc(ctx, op.o)
        d[:scaledesc] = desc(ctx, op.scale)
        op.stats === nothing || (d[:statsdesc] = desc(ctx, op.stats))
        op.seq_len_q === nothing || (d[:seq_len_qdesc] = desc(ctx, op.seq_len_q))
        op.seq_len_kv === nothing || (d[:seq_len_kvdesc] = desc(ctx, op.seq_len_kv))
        op.mask_subgraph === nothing ||
            lower_sdpa_mask_subgraph!(d, ctx, op.mask_subgraph)
    end
end

function lower(op::SDPABwdOp, ctx::LoweringContext)
    make_descriptor(:operation_sdpa_bwd) do d
        d[:qdesc] = desc(ctx, op.q)
        d[:kdesc] = desc(ctx, op.k)
        d[:vdesc] = desc(ctx, op.v)
        d[:odesc] = desc(ctx, op.o)
        d[:doddesc] = desc(ctx, op.dO)
        d[:statsdesc] = desc(ctx, op.stats)
        d[:scaledesc] = desc(ctx, op.scale)
        d[:dqdesc] = desc(ctx, op.dQ)
        d[:dkdesc] = desc(ctx, op.dK)
        d[:dvdesc] = desc(ctx, op.dV)
        op.seq_len_q === nothing || (d[:seq_len_qdesc] = desc(ctx, op.seq_len_q))
        op.seq_len_kv === nothing || (d[:seq_len_kvdesc] = desc(ctx, op.seq_len_kv))
        op.mask_subgraph === nothing ||
            lower_sdpa_mask_subgraph!(d, ctx, op.mask_subgraph)
    end
end

@public pointwise!, matmul!, reduction!, conv_fprop!, conv_dgrad!, conv_wgrad!,
        resample_fwd!, resample_bwd!, norm_fwd!, norm_bwd!, sdpa_fwd!, sdpa_bwd!
