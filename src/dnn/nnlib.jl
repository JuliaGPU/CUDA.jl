# interfacing with NNlib.jl

import ..CuArrays: CuVecOrMat, CuVector

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

function conv!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, cdims::DenseConvDims;
               alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionForward(y, x, w, cdims, alpha=alpha, algo=algo)
end

function ∇conv_filter!(dw::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                       cdims::DenseConvDims; alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionBackwardFilter(dw, x, dy, cdims, alpha=alpha, algo=algo)
end

function ∇conv_data!(dx::CuArray{T}, dy::CuArray{T}, w::CuArray{T},
                     cdims::DenseConvDims; alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionBackwardData(dx, w, dy, cdims, alpha=alpha, algo=algo)
end

∇conv_bias!(db::CuArray{T}, dy::CuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat =
  cudnnConvolutionBackwardBias(db, dy, alpha=alpha, beta=beta)

maxpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  cudnnPoolingForward(y, x, pdims; mode=0)

∇maxpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T},
          pdims::PoolDims) where T<:CUDNNFloat =
  cudnnPoolingBackward(dx, dy, x, y, pdims, mode=0)

meanpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  cudnnPoolingForward(y, x, pdims, mode=1)

∇meanpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T},
           pdims::PoolDims) where T<:CUDNNFloat =
  cudnnPoolingBackward(dx, dy, x, y, pdims, mode=1)
