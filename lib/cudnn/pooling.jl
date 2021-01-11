"""
    cudnnPoolingForward(x; mode, nanOpt, window, padding, stride, alpha)
    cudnnPoolingForward(x, d::cudnnPoolingDescriptor; alpha)
    cudnnPoolingForward!(y, x; mode, nanOpt, window, padding, stride, alpha, beta)
    cudnnPoolingForward!(y, x, d::cudnnPoolingDescriptor; alpha, beta)

Return pooled `x`, overwriting `y` if provided, according to keyword arguments or the
pooling descriptor `d`. Please see the [cuDNN
docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingForward) for
details.

The dimensions of `x,y` tensors that are less than 4-D are assumed to be padded on the left
with 1's. The first `n-2` are spatial dimensions, the last two are always assumed to be
channel and batch.

The arguments `window`, `padding`, and `stride` can be specified as `n-2` dimensional
vectors, tuples or a single integer which is assumed to be repeated `n-2` times. If any of
the entries is larger than the corresponding `x` dimension, the `x` dimension is used
instead.

Arguments:
* `mode = CUDNN_POOLING_MAX`: Pooling method, other options are `CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`, `CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`, `CUDNN_POOLING_MAX_DETERMINISTIC`
* `nanOpt = CUDNN_NOT_PROPAGATE_NAN`: NAN propagation policy, the other option is `CUDNN_PROPAGATE_NAN`
* `window = 2`: Pooling window size
* `padding = 0`: Padding assumed around `x`
* `stride = window`: How far to shift pooling window at each step
* `alpha=1, beta=0` can be used for scaling, i.e. `y .= alpha*op(x1) .+ beta*y`
"""
cudnnPoolingForward, cudnnPoolingForward!


# Public methods
cudnnPoolingForward(x; o...)     = cudnnPoolingForwardWithDefaults(x; o...)
cudnnPoolingForward!(y, x; o...) = cudnnPoolingForwardWithDefaults(x; y, o...)
cudnnPoolingForward(x, d::cudnnPoolingDescriptor; o...)     = cudnnPoolingForwardWithDefaults(x; poolingDesc=d, o...)
cudnnPoolingForward!(y, x, d::cudnnPoolingDescriptor; o...) = cudnnPoolingForwardWithDefaults(x; y, poolingDesc=d, o...)


# Private method
function cudnnPoolingForwardWithDefaults(
    x;                          # no type for x, could be AutoGrad.Value
    mode::cudnnPoolingMode_t = CUDNN_POOLING_MAX,
    nanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    window::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 2,
    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = window,
    poolingDesc::cudnnPoolingDescriptor = cudnnPoolingDescriptor(mode, nanOpt, Cint(max(2,ndims(x)-2)), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x))),
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
    y = cudnnPoolingForwardOutput(x, xDesc, poolingDesc, format),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y; format),
    alpha::Real = 1,
    beta::Real = 0,
)
    T = eltype(x)
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    cudnnPoolingForwardAD(x; poolingDesc, alpha, beta, xDesc, yDesc, y)
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


function cudnnPoolingForwardOutput(x, xDesc, poolingDesc, format)
    d = Array{Cint}(undef, max(4, ndims(x))) # d = [N,C,Yn,...,Y1] no matter what format
    cudnnGetPoolingNdForwardOutputDim(poolingDesc, xDesc, length(d), d)
    if length(d) > ndims(x) # This happens when x is (X,C,N), its TD is [N,C,X,1]
        @assert all(d[ndims(x)+1:end] .== 1)
        d = d[1:ndims(x)]
    end
    # ydims(NCHW)=(Y1,...,Yn,C,N) ydims(NHWC)=(C,Y1,...,Yn,N)
    ydims = (format === CUDNN_TENSOR_NCHW ? reverse(d) : (d[2],d[end:-1:3]...,d[1]))
    similar(x, ydims...)
end


# AD method
function cudnnPoolingForwardAD(x; poolingDesc, alpha, beta, xDesc, yDesc, y)
    cudnnPoolingForward(handle(), poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    return y
end


# Deprecated methods
function cudnnPoolingForward(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, pdims::NNlib.PoolDims;
                             alpha=1, beta=0, mode=CUDNN_POOLING_MAX) where {T,N}
    @warn "`cudnnPoolingForward(y,x,d::PoolDims)` is deprecated, please use one of the methods in `@doc cudnnPoolingForward`." maxlog=1
    cudnnPoolingForward!(y, x; window=NNlib.kernel_size(pdims), padding=nnlibPadding(pdims), stride=NNlib.stride(pdims), mode, alpha, beta)
end


