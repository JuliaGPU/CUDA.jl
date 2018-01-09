using NNlib
import NNlib: conv2d, softmax, softmax!, ∇softmax!

const CUDNNFloat = Union{Float16,Float32,Float64}
CUDNNArray{T<:CUDNNFloat,N} = CuArray{T,N}

CUDNNVector{T} = CUDNNArray{T,1}
CUDNNMatrix{T} = CUDNNArray{T,2}
CUDNNVecOrMat{T} = Union{CUDNNVector{T},CUDNNMatrix{T}}

function softmax!(out::CUDNNVecOrMat, xs::CUDNNVecOrMat)
  cudnnSoftmaxForward(xs, out)
  return out
end

function ∇softmax!(out::CUDNNVecOrMat, Δ::CUDNNVecOrMat, xs::CUDNNVecOrMat)
  cudnnSoftmaxBackward(softmax(xs), Δ, out)
  return out
end

function conv2d(x::CuArray{T,4}, w::CuArray{T,4};
                padding=0, stride=1, mode=0, alpha=1) where T<:CUDNNFloat
  y = similar(x, NNlib.cdims(w, x, padding=padding, stride=stride))
  cudnnConvolutionForward(y, x, w, padding=padding, stride=stride, mode=0, alpha=1)
end
