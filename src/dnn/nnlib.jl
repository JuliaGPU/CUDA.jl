using NNlib
import NNlib: conv2d, softmax, softmax!, ∇softmax!
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

function conv2d(x::CuArray{T,4}, w::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.cdims(w, x, padding=padding, stride=stride))
  cudnnConvolutionForward(y, x, w, padding=padding, stride=stride, mode=0, alpha=1)
end
