using NNlib
import NNlib: conv!, ∇conv_filter!, ∇conv_data!,
  maxpool!, meanpool!, ∇maxpool!, ∇meanpool!,
  softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax
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

function conv!(y::A, x::A, w::A;
               pad=0, stride=1, mode=0, alpha=1) where A<:CuArray{<:CUDNNFloat}
  cudnnConvolutionForward(y, x, w, padding=pad, stride=stride, mode=mode, alpha=alpha)
end

function ∇conv_filter!(dw::A, dy::A, x::A, w::A;
                       pad=0, stride=1, mode=0, alpha=1) where A<:CuArray{<:CUDNNFloat}
  cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=pad, stride=stride, mode=mode, alpha=alpha)
end

function ∇conv_data!(dx::A, dy::A, x::A, w::A;
                     pad=0, stride=1, mode=0, alpha=1) where A<:CuArray{<:CUDNNFloat}
  cudnnConvolutionBackwardData(dx, x, w, dy, padding=pad, stride=stride, mode=mode, alpha=alpha)
end


maxpool!(y::A, x::A, k; pad=map(_->0,k), stride=k) where A<:CuArray{<:CUDNNFloat} =
  cudnnPoolingForward(y, x, window=k, padding=pad, stride=stride, mode=0)

∇maxpool!(dx::A, dy::A, y::A, x::A, k;
          pad=map(_->0,k), stride=k) where A<:CuArray{<:CUDNNFloat} =
  cudnnPoolingBackward(dx, dy, x, y, window=k, padding=pad, stride=stride, mode=0)

meanpool!(y::A, x::A, k; pad=map(_->0,k), stride=k) where A<:CuArray{<:CUDNNFloat} =
  cudnnPoolingForward(y, x, window=k, padding=pad, stride=stride, mode=1)

∇meanpool!(dx::A, dy::A, y::A, x::A, k;
           pad=map(_->0,k), stride=k) where A<:CuArray{<:CUDNNFloat} =
  cudnnPoolingBackward(dx, dy, x, y, window=k, padding=pad, stride=stride, mode=1)
