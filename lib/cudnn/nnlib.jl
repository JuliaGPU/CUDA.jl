# interfacing with NNlib.jl

import NNlib: stride, padding, dilation, flipkernel, spatial_dims, kernel_size,
              conv!, conv_bias_act!, ∇conv_filter!, ∇conv_data!,
              maxpool!, meanpool!, ∇maxpool!, ∇meanpool!,
              softmax, softmax!, ∇softmax, ∇softmax!,
              logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

import DataStructures: DefaultDict


const CUDNNFloat = Union{Float16,Float32,Float64}

# Since CUDNN does not support 1D convolution, Conv in Flux will give a CUDNNError if the size is 1-dimensional.
fix1d(x) = x
fix1d(x::CuArray{T, 3}) where T = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
fix1d(cdims::DenseConvDims{1,K,C_in,C_out,S,P,D,F}) where {K,C_in,C_out,S,P,D,F} =
  DenseConvDims{2,(K...,1),C_in,C_out,(S...,1),(P...,0,0),(D...,1),F}((cdims.I...,1))
fix1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D,F} =
  PoolDims{2,(K...,1),(S...,1),(P...,0,0),(D...,1)}((pdims.I..., 1), pdims.C_in)

# We have to reshape the CuArray/PoolDims/DenseConvDims to 4D before feeding to CUDNN.
reshape4D(x::AbstractVector) = reshape(x, 1, 1, length(x), 1)
reshape4D(x::AbstractMatrix) = reshape(x, 1, 1, size(x)...)
reshape4D(x::AbstractArray{T,3}) where T = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
reshape4D(x::AbstractArray{T}) where T = x

workspacesize(x) = min(Mem.info()[1] ÷ 16, sizeof(x) * 2)

function perfChoose(perfResults, returnedAlgoCount)::UInt32
  if perfResults[1].status != 0
    return 0
  else
    (best_algo,best_time,best_memory) = (perfResults[1].algo,perfResults[1].time,perfResults[1].memory)
    for i = 2:returnedAlgoCount
      if perfResults[i].status == 0 && perfResults[i].memory < best_memory && perfResults[i].time < best_time * 1.1
        (best_algo,best_memory) = (perfResults[i].algo,perfResults[i].memory)
      end
    end
    return best_algo
  end
end


# Softmax

# in-place for x or dy
softmax(x::CuArray{T}; dims=1) where T<:CUDNNFloat =
  softmax!(x, x, dims=dims)

∇softmax(dy::CuArray{T}, x::CuArray{T}; dims=1) where T<:CUDNNFloat =
  ∇softmax!(dy, dy, x, dims=dims)

logsoftmax(x::CuArray{T}; dims=1) where T<:CUDNNFloat =
  logsoftmax!(x, x, dims=dims)

∇logsoftmax(dy::CuArray{T}, x::CuArray{T}; dims=1) where T<:CUDNNFloat =
  ∇logsoftmax!(dy, dy, x, dims=dims)

function softmax!(y::CuArray{T}, x::CuArray{T}; dims=1) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(x), reshape4D(y),
                      algo=CUDNN_SOFTMAX_FAST, mode=cudnnSoftmaxMode_t(dims-1))
  return y
end

function ∇softmax!(dx::CuArray{T}, dy::CuArray{T}, x::CuArray{T}; dims=1) where T<:CUDNNFloat
  y = softmax(x, dims=dims)
  cudnnSoftmaxBackward(reshape4D(y), reshape4D(dy), reshape4D(dx),
                       algo=CUDNN_SOFTMAX_FAST, mode=cudnnSoftmaxMode_t(dims-1))
  return dx
end

function logsoftmax!(y::CuArray{T}, x::CuArray{T}; dims=1) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(x), reshape4D(y),
                      algo=CUDNN_SOFTMAX_LOG, mode=cudnnSoftmaxMode_t(dims-1))
  return y
end

function ∇logsoftmax!(dx::CuArray{T}, dy::CuArray{T}, x::CuArray{T}; dims=1) where T<:CUDNNFloat
  y = logsoftmax(x, dims=dims)
  cudnnSoftmaxBackward(reshape4D(y), reshape4D(dy), reshape4D(dx),
                       algo=CUDNN_SOFTMAX_LOG, mode=cudnnSoftmaxMode_t(dims-1))
  return dx
end

# Convolution

const conv_forward_algos = DefaultDict{Tuple, Int32}(Int32(-1))
function conv!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, cdims::DenseConvDims;
               alpha=1, algo=-1) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  if algo < 0
    global conv_forward_algos
    key = (T, strides(x), strides(w), strides(y), cdims, size(x)[end])
    algo = conv_forward_algos[key]
    if algo < 0 # not in conv_forward_algos
      # algo = UInt32(cudnnGetConvolutionForwardAlgorithm(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), preference=2, workspacesize=workspacesize(x)) # will be removed in cuDNN 8
      # returnedAlgoCount, perfResults = cudnnGetConvolutionForwardAlgorithm_v7(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims))
      # returnedAlgoCount, perfResults = cudnnFindConvolutionForwardAlgorithm(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims))
      returnedAlgoCount, perfResults = cudnnFindConvolutionForwardAlgorithmEx(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), workspacesize=workspacesize(x))
      algo = perfChoose(perfResults, returnedAlgoCount)
      conv_forward_algos[key] = algo
    end
  end

  cudnnConvolutionForward(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), alpha=alpha, algo=algo)
  return y
end

function conv_bias_act!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, cdims::DenseConvDims, b::CuArray{T}, σ=identity;
                        z::CuArray{T}=y, alpha1=1, alpha2=0, algo=-1) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  if algo < 0
    global conv_forward_algos
    key = (T, strides(x), strides(w), strides(y), cdims, size(x)[end])
    algo = conv_forward_algos[key]
    if algo < 0 # not in conv_forward_algos
      # algo = UInt32(cudnnGetConvolutionForwardAlgorithm(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), preference=2, workspacesize=workspacesize(x))) # will be removed in cuDNN 8
      # returnedAlgoCount, perfResults = cudnnGetConvolutionForwardAlgorithm_v7(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims))
      # returnedAlgoCount, perfResults = cudnnFindConvolutionForwardAlgorithm(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims))
      returnedAlgoCount, perfResults = cudnnFindConvolutionForwardAlgorithmEx(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), workspacesize=workspacesize(x))
      algo = perfChoose(perfResults, returnedAlgoCount)
      conv_forward_algos[key] = algo
    end
  end

  # only relu and identity are supported
  if σ == NNlib.relu # always merge convolutions, bias, and relu, even when bias is turned off
    cudnnConvolutionBiasActivationForward(fix1d(y), fix1d(x), fix1d(w), fix1d(z), fix1d(b),
                                          fix1d(cdims), algo=algo, alpha1=alpha1, alpha2=alpha2,
                                          activationMode=CUDNN_ACTIVATION_RELU, activationCoeff=0.0)
  elseif algo == 1 && b != nothing # only merge convolution and bias if the fastest algorithm is also the only supported algorithm and the bias is not turned off
    # algo must be CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (1) when activationMode equals CUDNN_ACTIVATION_IDENTITY
    cudnnConvolutionBiasActivationForward(fix1d(y), fix1d(x), fix1d(w), fix1d(z), fix1d(b),
                                          fix1d(cdims), algo=algo, alpha1=alpha1, alpha2=alpha2,
                                          activationMode=CUDNN_ACTIVATION_IDENTITY, activationCoeff=0.0)
    σ.(y)
  else # fallback
    if b == nothing # bias is turned off
      σ.(conv!(y, x, w, cdims, alpha=alpha1, algo=algo))
    else # bias is turned on
      σ.(add_bias(conv!(y, x, w, cdims, alpha=alpha1, algo=algo), b))
    end
  end

  return y
end

const conv_data_algos = DefaultDict{Tuple, Int32}(Int32(-1))
function ∇conv_data!(dx::CuArray{T}, dy::CuArray{T}, w::CuArray{T},
                     cdims::DenseConvDims; alpha=1, algo=-1) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  if algo < 0
    global conv_data_algos
    key = (T, strides(dx), strides(w), strides(dy), cdims, size(dx)[end])
    algo = conv_data_algos[key]
    if algo < 0 # not in conv_data_algos
      # algo = UInt32(cudnnGetConvolutionBackwardDataAlgorithm(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims), preference=2, workspacesize=workspacesize(dx))) # will be removed in cuDNN 8
      # returnedAlgoCount, perfResults = cudnnGetConvolutionBackwardDataAlgorithm_v7(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims))
      # returnedAlgoCount, perfResults = cudnnFindConvolutionBackwardDataAlgorithm(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims))
      returnedAlgoCount, perfResults = cudnnFindConvolutionBackwardDataAlgorithmEx(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims), workspacesize=workspacesize(dx))
      algo = perfChoose(perfResults, returnedAlgoCount)
      conv_data_algos[key] = algo
    end
  end

  cudnnConvolutionBackwardData(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims), alpha=alpha, algo=algo)
  return dx
end

const conv_filter_algos = DefaultDict{Tuple, Int32}(Int32(-1))
function ∇conv_filter!(dw::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                       cdims::DenseConvDims; alpha=1, algo=-1) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  if algo < 0
    global conv_filter_algos
    # (type, batchsize, conv descriptor)
    key = (T, strides(x), strides(dw), strides(dy), cdims, size(x)[end])
    algo = conv_filter_algos[key]
    if algo < 0 # not in conv_filter_algos
      # algo = UInt32(cudnnGetConvolutionBackwardFilterAlgorithm(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims), preference=2, workspacesize=workspacesize(x))) # will be removed in cuDNN 8
      # returnedAlgoCount, perfResults = cudnnGetConvolutionBackwardFilterAlgorithm_v7(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims))
      # returnedAlgoCount, perfResults = cudnnFindConvolutionBackwardFilterAlgorithm(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims))
      returnedAlgoCount, perfResults = cudnnFindConvolutionBackwardFilterAlgorithmEx(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims), workspacesize=workspacesize(x))
      algo = perfChoose(perfResults, returnedAlgoCount)
      conv_filter_algos[key] = algo
    end
  end

  cudnnConvolutionBackwardFilter(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims), alpha=alpha, algo=algo)
  return dw
end


# Bias

# in-place for x (add b to x)
add_bias(x::CuArray{T}, b::CuArray{T})  where {T<:CUDNNFloat} =
  (cudnnAddTensor(reshape4D(x), reshape4D(b)); return x)

∇conv_bias!(db::CuArray{T}, dy::CuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat =
  (cudnnConvolutionBackwardBias(fix1d(db), fix1d(dy), alpha=alpha, beta=beta); return db)


# Pooling

maxpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingForward(fix1d(y), fix1d(x), fix1d(pdims); mode=0); return y)

∇maxpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingBackward(fix1d(dx), fix1d(dy), fix1d(x), fix1d(y), fix1d(pdims), mode=0); return dx)

meanpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingForward(fix1d(y), fix1d(x), fix1d(pdims), mode=1); return y)

∇meanpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingBackward(fix1d(dx), fix1d(dy), fix1d(x), fix1d(y), fix1d(pdims), mode=1); return dx)


# Activation

# in-place for x
Base.broadcasted(::typeof(NNlib.σ), x::CuArray{T}) where {T<:CUDNNFloat} =
  (cudnnActivationForward(reshape4D(x), mode=CUDNN_ACTIVATION_SIGMOID, coeff=0.0); return x)

Base.broadcasted(::typeof(NNlib.relu), x::CuArray{T}) where {T<:CUDNNFloat} =
  (cudnnActivationForward(reshape4D(x), mode=CUDNN_ACTIVATION_RELU, coeff=0.0); return x)

Base.broadcasted(::typeof(NNlib.tanh), x::CuArray{T}) where {T<:CUDNNFloat} =
  (cudnnActivationForward(reshape4D(x), mode=CUDNN_ACTIVATION_TANH, coeff=0.0); return x)

Base.broadcasted(::typeof(NNlib.relu6), x::CuArray{T}) where {T<:CUDNNFloat} =
  (cudnnActivationForward(reshape4D(x), mode=CUDNN_ACTIVATION_CLIPPED_RELU, coeff=6.0); return x)

Base.broadcasted(::typeof(NNlib.elu), x::CuArray{T}) where {T<:CUDNNFloat} =
  (cudnnActivationForward(reshape4D(x), mode=CUDNN_ACTIVATION_ELU, coeff=1.0); return x)

# CUDNN_ACTIVATION_IDENTITY does not work with cudnnActivationForward
Base.broadcasted(::typeof(NNlib.identity), x::CuArray{T}) where {T<:CUDNNFloat} = x

Base.broadcasted(::typeof(NNlib.leakyrelu), x::CuArray{T}, a=T(0.01)) where {T<:CUDNNFloat} =
  (cudnnOpTensor(CUDNN_OP_TENSOR_MAX, reshape4D(x), reshape4D(x), reshape4D(x), alpha1=a); return x)
