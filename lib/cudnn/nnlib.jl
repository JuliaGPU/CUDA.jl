# interfacing with NNlib.jl

import NNlib: conv!, ∇conv_filter!, ∇conv_data!, stride, dilation, flipkernel,
  maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, spatial_dims, padding, kernel_size,
  softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax


# Softmax

const CUDNNFloat = Union{Float16,Float32,Float64}

reshape4D(x::AbstractVector) = reshape(x, 1, 1, length(x), 1)
reshape4D(x::AbstractMatrix) = reshape(x, 1, 1, size(x)...)

function softmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(xs), reshape4D(out))
  return out
end

function ∇softmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(reshape4D(softmax(xs)), reshape4D(Δ), reshape4D(out))
  return out
end

function logsoftmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(xs), reshape4D(out), algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

function ∇logsoftmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(reshape4D(logsoftmax(xs)), reshape4D(Δ), reshape4D(out);
                       algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

∇logsoftmax(Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat =
  ∇logsoftmax!(similar(xs), Δ, xs)


# Convolution

# Since CUDNN does not support 1D convolution, Conv in Flux will give a CUDNNError if the size is 1-dimensional.
# We have to reshape the CuArray/PoolDims/DenseConvDims to 4D before feeding to CUDNN.
fix1d(x) = x

fix1d(x::CuArray{T, 3}) where T = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))

fix1d(cdims::DenseConvDims{1,K,C_in,C_out,S,P,D,F}) where {K,C_in,C_out,S,P,D,F} =
  DenseConvDims{2,(K...,1),C_in,C_out,(S...,1),(P...,0,0),(D...,1),F}((cdims.I...,1))

fix1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D,F} =
  PoolDims{2,(K...,1),(S...,1),(P...,0,0),(D...,1)}((pdims.I..., 1), pdims.C_in)

function conv!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, cdims::DenseConvDims;
               alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end
  cudnnConvolutionForward(fix1d(y), fix1d(x), fix1d(w), fix1d(cdims), alpha=alpha, algo=algo)
  return y
end

function ∇conv_filter!(dw::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                       cdims::DenseConvDims; alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionBackwardFilter(fix1d(dw), fix1d(x), fix1d(dy), fix1d(cdims), alpha=alpha, algo=algo)
  return dw
end

function ∇conv_data!(dx::CuArray{T}, dy::CuArray{T}, w::CuArray{T},
                     cdims::DenseConvDims; alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionBackwardData(fix1d(dx), fix1d(w), fix1d(dy), fix1d(cdims), alpha=alpha, algo=algo)
  return dx
end

∇conv_bias!(db::CuArray{T}, dy::CuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat =
  (cudnnConvolutionBackwardBias(fix1d(db), fix1d(dy), alpha=alpha, beta=beta); return db)

maxpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingForward(fix1d(y), fix1d(x), fix1d(pdims); mode=0); return y)

∇maxpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T},
          pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingBackward(fix1d(dx), fix1d(dy), fix1d(x), fix1d(y), fix1d(pdims), mode=0); return dx)

meanpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingForward(fix1d(y), fix1d(x), fix1d(pdims), mode=1); return y)

∇meanpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T},
           pdims::PoolDims) where T<:CUDNNFloat =
  (cudnnPoolingBackward(fix1d(dx), fix1d(dy), fix1d(x), fix1d(y), fix1d(pdims), mode=1); return dx)
