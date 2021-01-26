# interfacing with NNlib.jl

import NNlib: stride, padding, dilation, flipkernel, spatial_dims, kernel_size,
              conv!, ∇conv_filter!, ∇conv_data!,
              maxpool!, meanpool!, ∇maxpool!, ∇meanpool!,
              softmax, softmax!, ∇softmax, ∇softmax!,
              logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

import DataStructures: DefaultDict


const CUDNNFloat = Union{Float16,Float32,Float64}

# Since CUDNN does not support 1D convolution, Conv in Flux will give a CUDNNError if the size is 1-dimensional.
fix1d(x) = x
fix1d(x::DenseCuArray{T, 3}) where T = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
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
softmax(x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat =
  softmax!(x, x, dims=dims)

∇softmax(dy::DenseCuArray{T}, x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat =
  ∇softmax!(dy, dy, x, dims=dims)

logsoftmax(x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat =
  logsoftmax!(x, x, dims=dims)

∇logsoftmax(dy::DenseCuArray{T}, x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat =
  ∇logsoftmax!(dy, dy, x, dims=dims)

function softmax!(y::DenseCuArray{T}, x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(x), reshape4D(y),
                      algo=CUDNN_SOFTMAX_FAST, mode=cudnnSoftmaxMode_t(dims-1))
  return y
end

function ∇softmax!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat
  y = softmax(x, dims=dims)
  cudnnSoftmaxBackward(reshape4D(y), reshape4D(dy), reshape4D(dx),
                       algo=CUDNN_SOFTMAX_FAST, mode=cudnnSoftmaxMode_t(dims-1))
  return dx
end

function logsoftmax!(y::DenseCuArray{T}, x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(x), reshape4D(y),
                      algo=CUDNN_SOFTMAX_LOG, mode=cudnnSoftmaxMode_t(dims-1))
  return y
end

function ∇logsoftmax!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, x::DenseCuArray{T}; dims=1) where T<:CUDNNFloat
  y = logsoftmax(x, dims=dims)
  cudnnSoftmaxBackward(reshape4D(y), reshape4D(dy), reshape4D(dx),
                       algo=CUDNN_SOFTMAX_LOG, mode=cudnnSoftmaxMode_t(dims-1))
  return dx
end

# Convolution

const conv_forward_algos = DefaultDict{Tuple, Int32}(Int32(-1))
function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims;
               algo=-1, alpha=1, kw...) where T<:CUDNNFloat
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

  cudnnConvolutionForward(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), algo=algo; alpha, kw...)
  return y
end

if isdefined(NNlib, :conv_bias_act!)
function NNlib.conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims, b::DenseCuArray{T}, σ=identity;
                              z::DenseCuArray{T}=y, alpha1=1, alpha2=0, algo=-1) where T<:CUDNNFloat
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
end

const conv_data_algos = DefaultDict{Tuple, Int32}(Int32(-1))
function ∇conv_data!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, w::DenseCuArray{T},
                     cdims::DenseConvDims; algo=-1, alpha=1, kw...) where T<:CUDNNFloat
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

  cudnnConvolutionBackwardData(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims), algo=algo; alpha, kw...)
  return dx
end

const conv_filter_algos = DefaultDict{Tuple, Int32}(Int32(-1))
function ∇conv_filter!(dw::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                       cdims::DenseConvDims; algo=-1, alpha=1, kw...) where T<:CUDNNFloat
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

  cudnnConvolutionBackwardFilter(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims), algo=algo; alpha, kw...)
  return dw
end


# Bias

# in-place for x (add b to x)
add_bias(x::DenseCuArray{T}, b::DenseCuArray{T})  where {T<:CUDNNFloat} =
  (cudnnAddTensor(reshape4D(x), reshape4D(b)); return x)

∇conv_bias!(db::DenseCuArray{T}, dy::DenseCuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat =
  (cudnnConvolutionBackwardBias(fix1d(db), fix1d(dy), alpha=alpha, beta=beta); return db)


# Pooling

maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingForward(fix1d(y), fix1d(x), fix1d(pdims); mode=0); return y)

∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingBackward(fix1d(dx), fix1d(dy), fix1d(x), fix1d(y), fix1d(pdims), mode=0); return dx)

meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingForward(fix1d(y), fix1d(x), fix1d(pdims), mode=1); return y)

∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingBackward(fix1d(dx), fix1d(dy), fix1d(x), fix1d(y), fix1d(pdims), mode=1); return dx)


# Activation

using Base.Broadcast

for (f, op) in [
  CUDA.tanh       => (src,dst)->cudnnActivationForward(reshape4D(src), reshape4D(dst),
                                                       mode=CUDNN_ACTIVATION_TANH),
  NNlib.σ         => (src,dst)->cudnnActivationForward(reshape4D(src), reshape4D(dst),
                                                       mode=CUDNN_ACTIVATION_SIGMOID),
  NNlib.elu       => (src,dst)->cudnnActivationForward(reshape4D(src), reshape4D(dst),
                                                       mode=CUDNN_ACTIVATION_ELU),
  NNlib.relu      => (src,dst)->cudnnActivationForward(reshape4D(src), reshape4D(dst),
                                                       mode=CUDNN_ACTIVATION_RELU),
  NNlib.relu6     => (src,dst)->cudnnActivationForward(reshape4D(src), reshape4D(dst),
                                                       mode=CUDNN_ACTIVATION_CLIPPED_RELU,
                                                       coeff=6.0),
  NNlib.leakyrelu => (src,dst)->cudnnOpTensor(CUDNN_OP_TENSOR_MAX, reshape4D(src),
                                              reshape4D(src), reshape4D(dst),
                                              alpha1=0.01)]
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
