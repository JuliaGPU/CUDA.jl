using CUDNN
import NNlib: softmax!

const CUDNNFloat = Union{Float16,Float32,Float64}
CUDNNArray{T<:CUDNNFloat,N} = CuArray{T,N}
CUDNNVector{T<:CUDNNFloat} = CuArray{T,1}
CUDNNMatrix{T<:CUDNNFloat} = CuArray{T,2}
CUDNNVecOrMat{T} = Union{CUDNNVector{T},CUDNNMatrix{T}}

function softmax!(out::CUDNNVecOrMat, xs::CUDNNVecOrMat)
  cudnnSoftmaxForward(CUDAdrv.CuArray.((xs, out))...)
  return out
end
