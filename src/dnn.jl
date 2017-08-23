using CUDNN
import NNlib: softmax, softmax!, ∇softmax!

const CUDNNFloat = Union{Float16,Float32,Float64}
CUDNNArray{T<:CUDNNFloat,N} = CuArray{T,N}

CUDNNVector{T} = CUDNNArray{T,1}
CUDNNMatrix{T} = CUDNNArray{T,2}
CUDNNVecOrMat{T} = Union{CUDNNVector{T},CUDNNMatrix{T}}

function softmax!(out::CUDNNVecOrMat, xs::CUDNNVecOrMat)
  cudnnSoftmaxForward(CUDAdrv.CuArray.((xs, out))...)
  return out
end

function ∇softmax!(out::CUDNNVecOrMat, Δ::CUDNNVecOrMat, xs::CUDNNVecOrMat)
  cudnnSoftmaxBackward(CUDAdrv.CuArray.((softmax(xs), Δ, out))...)
  return out
end
