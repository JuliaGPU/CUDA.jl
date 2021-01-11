# interfacing with NNlib.jl

import NNlib: stride, padding, dilation, flipkernel, spatial_dims, kernel_size,
    conv!, ∇conv_filter!, ∇conv_data!,
    maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, PoolDims,
    softmax, softmax!, ∇softmax, ∇softmax!,
    logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

import DataStructures: DefaultDict


const CUDNNFloat = Union{Float16,Float32,Float64}

# Since CUDNN does not support 1D convolution, Conv in Flux will give a CUDNNError if the size is 1-dimensional.
fix1d(x) = x
fix1d(x::DenseCuArray{T, 3}) where T = reshape(x, 1, size(x, 1), size(x, 2), size(x, 3))
fix1d(cdims::DenseConvDims{1,K,C_in,C_out,S,P,D,F}) where {K,C_in,C_out,S,P,D,F} =
    DenseConvDims{2,(1,K...),C_in,C_out,(1,S...),(0,0,P...),(1,D...),F}((1,cdims.I...))
fix1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D,F} =
    PoolDims{2,(1,K...),(1,S...),(0,0,P...),(1,D...)}((1,pdims.I...), pdims.C_in)

# Softmax

# @denizyuret: do not do inplace operations with softmax/logsoftmax when (1) cpu version is not, (2) one can use softmax!
function softmax(x::T; dims=1) where {T<:DenseCuArray}
    softmax!(similar(x), x; dims)
end

function ∇softmax(dy::T, x::T, y::T; dims=1) where {T<:DenseCuArray}
    ∇softmax!(similar(x), dy, x, y; dims)
end

function logsoftmax(x::T; dims=1) where {T<:DenseCuArray}
    logsoftmax!(similar(x), x; dims)
end

function ∇logsoftmax(dy::T, x::T, y::T; dims=1) where {T<:DenseCuArray}
    ∇logsoftmax!(similar(x), dy, x, y; dims)
end


# @denizyuret: recalculating y in ∇softmax! is a big waste, the nnlib API should be changed:
function ∇softmax(dy::T, x::T; dims=1) where {T<:DenseCuArray}
    @warn "∇softmax(dy,x) should be deprecated, please use ∇softmax(dy,x,y)" maxlog=1
    ∇softmax!(similar(x), dy, x, softmax(x); dims)
end

function ∇softmax!(dx::T, dy::T, x::T; dims=1) where {T<:DenseCuArray}
    @warn "∇softmax!(dx,dy,x) should be deprecated, please use ∇softmax!(dx,dy,x,y)" maxlog=1
    ∇softmax!(dx, dy, x, softmax(x); dims)
end

function ∇logsoftmax(dy::T, x::T; dims=1) where {T<:DenseCuArray}
    @warn "∇logsoftmax(dy,x) should be deprecated, please use ∇logsoftmax(dy,x,y)" maxlog=1
    ∇logsoftmax!(similar(x), dy, x, logsoftmax(x); dims)
end

function ∇logsoftmax!(dx::T, dy::T, x::T; dims=1) where {T<:DenseCuArray}
    @warn "∇logsoftmax!(dx,dy,x) should be deprecated, please use ∇logsoftmax!(dx,dy,x,y)" maxlog=1
    ∇logsoftmax!(dx, dy, x, logsoftmax(x); dims)
end


# @denizyuret: backup implementations for unsupported/slow size/dims combinations:
function _softmax!(y::T, x::T; dims) where {T<:DenseCuArray}
    y .= exp.(x .- maximum(x; dims))
    y ./= sum(y; dims)
end

function _∇softmax!(dx::T, dy::T, x::T, y::T; dims) where {T<:DenseCuArray}
    dx .= y .* (dy .- sum(dy .* y; dims))
end

function _logsoftmax!(y::T, x::T; dims) where {T<:DenseCuArray}
    y .= x .- maximum(x; dims)
    y .-= log.(sum(exp.(y); dims))
end

function _∇logsoftmax!(dx::T, dy::T, x::T, y::T; dims) where {T<:DenseCuArray}
    dx .= dy .- sum(dy; dims) .* exp.(y)
end

# Trick by @norci to use cudnn for softmax dims args that are contiguous: 
# If dims=(dmin:dmax) then CUDNN_SOFTMAX_MODE_CHANNEL does the trick with reshape 
#    (1, prod(size(x)[1:dmin-1]), prod(size(x)[dmin:dmax]), :)
# softmaxdims returns nothing when the backup implementation should be used.

function softmaxdims(x, dims)
    dims === Colon() && return (1, 1, length(x), 1)
    mind,maxd = minimum(dims),maximum(dims)
    all(i in dims for i in mind:maxd) || return nothing # cannot handle if not contiguous
    stride = dimsize = 1
    for i in 1:(mind-1); stride *= size(x,i); end # Using size(x,i) assumes trailing dims = 1, robust to maxd > ndims(x)
    for i in mind:maxd; dimsize *= size(x,i); end
    batchsize = length(x)÷(stride*dimsize)
    # Here is a region where cudnn is slower, so we go with the backup:
    batchsize == 1 && 64 <= stride <= 4096 && 64 <= dimsize <= 4096 && return nothing
    return (1, stride, dimsize, batchsize)
end

# Determine softmax algo based on math_mode

softmaxalgo() = (CUDA.math_mode()===CUDA.FAST_MATH ? CUDNN_SOFTMAX_FAST : CUDNN_SOFTMAX_ACCURATE)

# Main implementations:

function softmax!(y::T, x::T = y; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return _softmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo = softmaxalgo())
    return y
end

function ∇softmax!(dx::T, dy::T, x::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(x, dims)
    s === nothing && return _∇softmax!(dx, dy, x, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(x,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), softmaxalgo(), CUDNN_SOFTMAX_MODE_CHANNEL, 
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end

function logsoftmax!(y::T, x::T = y; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return _logsoftmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo = CUDNN_SOFTMAX_LOG)
    return y
end

function ∇logsoftmax!(dx::T, dy::T, x::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(x, dims)
    s === nothing && return _∇logsoftmax!(dx, dy, x, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(x,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, 
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end


# Convolution

function cudnnConvolutionDescriptor(cdims::DenseConvDims, x::DenseCuArray{T}) where T
    cdims, x = fix1d(cdims), fix1d(x)
    mode=(NNlib.flipkernel(cdims) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION)
    cudnnConvolutionDescriptor(convdims(nnlibPadding(cdims),size(x)), convdims(NNlib.stride(cdims),size(x)), convdims(NNlib.dilation(cdims),size(x)), mode, cudnnDataType(T), math_mode(), CUDNN_DEFAULT_REORDER, Cint(1))
end

function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims;
               alpha=1, algo=-1) where T<:CUDNNFloat
    if version() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    d = cudnnConvolutionDescriptor(cdims, x)
    cudnnConvolutionForward!(y, w, x, d)
end

if isdefined(NNlib, :conv_bias_act!)
    function NNlib.conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims, b::DenseCuArray{T}, σ=identity;
                                  z::DenseCuArray{T}=y, alpha1=1, alpha2=0, algo=-1) where T<:CUDNNFloat
        if version() < v"6"
            all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
        end
        if algo != -1
            @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
        end    
        d = cudnnConvolutionDescriptor(cdims, x)
        # only relu and identity are supported by cudnnConvolutionForward!
        activation = (σ == NNlib.relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY)
        cudnnConvolutionForward!(y, w, x, d; z, bias=b, activation, alpha=alpha1, beta=alpha2)
        if σ != NNlib.relu && σ != identity
            y = σ.(y)
        end
        return y
    end
end

function ∇conv_data!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, w::DenseCuArray{T},
                     cdims::DenseConvDims; alpha=1, algo=-1) where T<:CUDNNFloat
    if version() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end    
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,0);
    xDesc, yDesc, wDesc = cudnnTensorDescriptor(dx), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(w)
    convDesc = cudnnConvolutionDescriptor(cdims, dx)
    p = cudnnConvolutionBwdDataAlgoPerf(wDesc, w, yDesc, dy, convDesc, xDesc, dx)
    @workspace size=p.memory workspace->cudnnConvolutionBackwardData(handle(), alpha, wDesc, w, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta, xDesc, dx)
    return dx
end

function ∇conv_filter!(dw::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                       cdims::DenseConvDims; alpha=1, algo=-1) where T<:CUDNNFloat
    if version() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end    
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,0);
    xDesc, yDesc, wDesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(dw)
    convDesc = cudnnConvolutionDescriptor(cdims, x)
    p = cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, yDesc, dy, convDesc, wDesc, dw);
    @workspace size=p.memory workspace->cudnnConvolutionBackwardFilter(handle(), alpha, xDesc, x, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta, wDesc, dw);
    return dw
end



# Bias

# in-place for x (add b to x)
# @denizyuret: cudnnAddTensor only supports (a,b,c,d)+(1,1,c,1) and (a,b,c,d,e)+(1,1,1,d,1), use cudnnOpTensor instead.
# Compared to libknet8 x .+ b it is ~2x slower for (1,1,100,100), ~30% faster for (14,14,256,32)
# CUDA.jl x .+ b is 2x slower than both
add_bias(x::DenseCuArray{T}, b::DenseCuArray{T})  where {T<:CUDNNFloat} =
    (cudnnAddTensor!(x, b); return x)

function ∇conv_bias!(db::DenseCuArray{T}, dy::DenseCuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat
    alpha,beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    bDesc, yDesc = cudnnTensorDescriptor.((db,dy))
    cudnnConvolutionBackwardBias(handle(), alpha, yDesc, dy, beta, bDesc, db)
    return db
end

# Pooling

function cudnnPoolingDescriptor(pdims::PoolDims, x::DenseCuArray{T}, mode::cudnnPoolingMode_t) where T
    pdims, x = fix1d(pdims), fix1d(x)
    window, padding, stride = NNlib.kernel_size(pdims), nnlibPadding(pdims), NNlib.stride(pdims)
    nanOpt = CUDNN_NOT_PROPAGATE_NAN
    cudnnPoolingDescriptor(mode, nanOpt, Cint(max(2,ndims(x)-2)), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x)))
end

function maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_MAX)
    cudnnPoolingForward!(y, x, d)
end

function ∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_MAX)
    alpha, beta = scalingParameter(T,1), scalingParameter(T,0)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    cudnnPoolingForward!(y, x, d)
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    alpha, beta = scalingParameter(T,1), scalingParameter(T,0)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

# Activation

using Base.Broadcast

for (f, op) in [
    CUDA.tanh       => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_TANH),
    NNlib.σ         => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_SIGMOID),
    NNlib.elu       => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_ELU),
    NNlib.relu      => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_RELU),
    NNlib.relu6     => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_CLIPPED_RELU, coef=6.0),
    NNlib.leakyrelu => (src,dst)->cudnnOpTensor!(dst, src, src; op=CUDNN_OP_TENSOR_MAX, alpha1=0.01)]
    @eval begin
        # in-place
        function Base.materialize!(dst::DenseCuArray{<:CUDNNFloat},
                                   bc::Broadcast.Broadcasted{<:Any,<:Any,typeof($f),<:Tuple{DenseCuArray}})
            $op(bc.args[1], dst)
            return dst
        end

        # out of place
        function Base.materialize(bc::Broadcast.Broadcasted{<:Any,<:Any,typeof($f),<:Tuple{DenseCuArray}})
            ElType = Broadcast.combine_eltypes(bc.f, bc.args)
            dst = similar(bc, ElType)
            $op(bc.args[1], dst)
            return dst
        end
    end
end

# CUDNN_ACTIVATION_IDENTITY does not work with cudnnActivationForward
# FIXME: put this optimization in GPUArrays' `copyto!` (like Base.Broadcast's `copyto!`)
Base.broadcasted(::typeof(identity), x::DenseCuArray{T}) where {T<:CUDNNFloat} = x


# Compatibility shims until users upgrade to new NNlib format
function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}; pad=0, stride=1, flipkernel=0, dilation=1, kwargs...) where {T<:CUDNNFloat}
    cdims = DenseConvDims(x, w; padding=pad, stride=stride, flipkernel=(flipkernel!=0), dilation=dilation)
    return conv!(y, x, w, cdims; kwargs...)
end

function ∇conv_filter!(dw::DenseCuArray{T}, dy::DenseCuArray{T}, x::DenseCuArray{T}; pad=0, stride=1, flipkernel=0, dilation=1, kwargs...) where {T<:CUDNNFloat}
    cdims = DenseConvDims(x, dw; padding=pad, stride=stride, flipkernel=(flipkernel!=0), dilation=dilation)
    # NOTE!!!  This compat shim re-arranges the argument order!
    return ∇conv_filter!(dw, x, dy, cdims; kwargs...)
end

function maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, k; pad=map(_->0,k), stride=k) where {T<:CUDNNFloat}
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return maxpool!(y, x, pdims)
end

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, k; pad=map(_->0,k), stride=k) where {T<:CUDNNFloat}
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return meanpool!(y, x, pdims)
end
