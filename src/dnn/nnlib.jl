using NNlib
import NNlib: conv2d, conv2d_grad_x, conv2d_grad_w, pool, pool_grad, softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax
using ..CuArrays: CuVecOrMat

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
  cudnnSoftmaxBackward(reshape4D(logsoftmax(xs)), reshape4D(Δ), reshape4D(out), algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

∇logsoftmax(Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat = ∇logsoftmax!(similar(xs), Δ, xs)

function conv2d(x::CuArray{T,4}, w::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.cdims(w, x, padding=padding, stride=stride))
  cudnnConvolutionForward(y, x, w, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv3d(x::CuArray{T,5}, w::CuArray{T,5};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.cdims(w, x, padding=padding, stride=stride))
  cudnnConvolutionForward(y, x, w, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv2d_grad_w(x::CuArray{T,4}, w::CuArray{T,4}, dy::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  dw = similar(w)
  cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv3d_grad_w(x::CuArray{T,5}, w::CuArray{T,5}, dy::CuArray{T,5};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  dw = similar(w)
  cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv2d_grad_x(x::CuArray{T,4}, w::CuArray{T,4}, dy::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  dx = similar(x)
  cudnnConvolutionBackwardData(dx, x, w, dy, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function conv3d_grad_x(x::CuArray{T,5}, w::CuArray{T,5}, dy::CuArray{T,5};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  dx = similar(x)
  cudnnConvolutionBackwardData(dx, x, w, dy, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function pool2d(x::CuArray{T,4}; window=2, padding=0, stride=window, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.pdims(x))
  cudnnPoolingForward(y, x, window=window, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function pool3d(x::CuArray{T,5}; window=2, padding=0, stride=window, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.pdims(x))
  cudnnPoolingForward(y, x, window=window, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function pool2d_grad(x::CuArray{T,4}, y::CuArray{T,4}, dy::CuArray{T,4};
                   window=2, padding=0, stride=window, mode=0, alpha=1) where T<:CUDNNFloat
  dx = similar(x)
  cudnnPoolingBackward(dx, dy, x, y, window=window, padding=padding, stride=stride, mode=mode, alpha=alpha)
end

function pool3d_grad(x::CuArray{T,5}, y::CuArray{T,5}, dy::CuArray{T,5};
                   window=2, padding=0, stride=window, mode=0, alpha=1) where T<:CUDNNFloat
  dx = similar(x)
  cudnnPoolingBackward(dx, dy, x, y, window=window, padding=padding, stride=stride, mode=mode, alpha=alpha)
end
