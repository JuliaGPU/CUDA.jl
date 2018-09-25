using NNlib
import NNlib: conv!, ∇conv_filter!, ∇conv_data!,
  maxpool!, meanpool!, ∇maxpool!, ∇meanpool!,
  softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax
import ..CuArrays: CuVecOrMat, @cufunc, cufunc, CuVector
using CUDAnative

# Activation functions
@cufunc σ(x) = ifelse(x < -80, zero(x), one(x) / (one(x) + exp(-x)))

@cufunc function logσ(x)
  max_v = max(zero(x), -x)
  z = exp(-max_v) + exp(-x-max_v)
  -(max_v + log(z))
end

@cufunc elu(x, α = one(x)) =
  ifelse(x ≥ 0, x/1, α * (exp(x) - one(x)))

# TODO: make @cufunc recognise its own definitions
cufunc(::typeof(swish)) = x -> x * cufunc(σ)(x)

@cufunc function selu(x)
  λ = oftype(x/1, 1.0507009873554804934193349852946)
  α = oftype(x/1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, x/1, α * (exp(x) - 1))
end

@cufunc softplus(x) = log1p(exp(x))

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
  cudnnSoftmaxBackward(reshape4D(logsoftmax(xs)), reshape4D(Δ), reshape4D(out), algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

∇logsoftmax(Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat = ∇logsoftmax!(similar(xs), Δ, xs)

# Convolution

const _conv_workspace = Ref{CuVector{UInt8}}()

function conv_workspace(bytes)
  if isassigned(_conv_workspace) && bytes <= length(_conv_workspace[])
    _conv_workspace[]
  else
    _conv_workspace[] = CuVector{UInt8}(bytes)
  end
end

function conv!(y::A, x::A, w::A;
               pad = 0, stride = 1, mode = 0,
               alpha = 1, dilation = 1, workspace::Union{CuVector, Nothing} = nothing,
               algo = 0) where A<:CuArray{<:CUDNNFloat}
  all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in CuArrays")
  if workspace === nothing
    workspace_size = cudnnGetConvolutionForwardWorkspaceSize(y, x, w, padding = pad, stride = stride, algo = algo, mode = mode)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionForward(y, x, w, padding=pad, stride=stride, mode=mode, alpha=alpha, algo = algo,
                          workspace=workspace, workspace_size=workspace_size)
end

function ∇conv_filter!(dw::A, dy::A, x::A, w::A;
                       pad = 0, stride = 1, mode = 0,
                       alpha = 1, dilation = 1, workspace::Union{CuVector, Nothing} = nothing,
                       algo = 0) where A<:CuArray{<:CUDNNFloat}
  all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in CuArrays")
  if workspace === nothing
    workspace_size = cudnnGetConvolutionBackwardFilterWorkspaceSize(dw, x, w, dy, padding = pad, stride = stride, algo = algo, mode = mode)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=pad, stride=stride, mode=mode, alpha=alpha, algo = algo,
                                 workspace=workspace, workspace_size=workspace_size)
end

function ∇conv_data!(dx::A, dy::A, x::A, w::A;
                     pad = 0, stride = 1, mode = 0,
                     alpha = 1, dilation = 1, workspace::Union{CuVector, Nothing} = nothing,
                     algo = 0) where A<:CuArray{<:CUDNNFloat}
  all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in CuArrays")
  if workspace === nothing
    workspace_size = cudnnGetConvolutionBackwardDataWorkspaceSize(dx, x, w, dy, padding = pad, stride = stride, algo = algo, mode = mode)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionBackwardData(dx, x, w, dy, padding=pad, stride=stride, mode=mode, alpha=alpha, algo = algo,
                               workspace=workspace, workspace_size=workspace_size)
end

function ∇conv_bias!(db::A, dy::A; alpha = 1, beta = 0) where A<:CuArray{<:CUDNNFloat}
  cudnnConvolutionBackwardBias(db, dy, alpha=alpha, beta=beta)
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
