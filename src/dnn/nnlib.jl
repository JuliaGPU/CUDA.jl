using NNlib
import NNlib: conv2d, conv2d_grad_x, conv2d_grad_w, pool, pool_grad, softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax
using ..CuArrays: CuVecOrMat

const CUDNNFloat = Union{Float16,Float32,Float64}

function softmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(xs, out)
  return out
end

function ∇softmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(softmax(xs), Δ, out)
  return out
end

function logsoftmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(xs, out, algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

function ∇logsoftmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(logsoftmax(xs), Δ, out, algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

∇logsoftmax(Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat = ∇logsoftmax!(similar(xs), Δ, xs)

function conv2d(x::CuArray{T,4}, w::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.cdims(w, x, padding=padding, stride=stride))
  cudnnConvolutionForward(y, x, w, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv2d_grad_w(x::CuArray{T,4}, w::CuArray{T,4}, dy::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  dw = similar(w)
  cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv2d_grad_x(x::CuArray{T,4}, w::CuArray{T,4}, dy::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  dx = similar(x)
  cudnnConvolutionBackwardData(dx, x, w, dy, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function pool(x::CuArray{T,4}; window=2, padding=0, stride=window, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.pdims(x))
  cudnnPoolingForward(y, x, window=window, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function pool_grad(x::CuArray{T,4}, y::CuArray{T,4}, dy::CuArray{T,4};
                   window=2, padding=0, stride=window, mode=0, alpha=1) where T<:CUDNNFloat
  dx = similar(x)
  cudnnPoolingBackward(dx, dy, x, y, window=window, padding=padding, stride=stride, mode=mode, alpha=alpha)
end
