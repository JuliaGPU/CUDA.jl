# CUDNN_RNN_RELU: Stock RNN with ReLu activation
# CUDNN_RNN_TANH: Stock RNN with tanh activation
# CUDNN_LSTM:     LSTM with no peephole connections
# CUDNN_GRU:      Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1)

# param layout:
# RNN: [weight, bias] × [input, hidden]
# GRU: [weight, bias] × [input, hidden] × [reset, update, newmem]
# LSTM: [weight, bias] × [input, hidden] × [input, forget, newmem, output]

using LinearAlgebra: copy_transpose!

function params(w::CuVector, input, hidden, n = 1)
  slice(offset, shape) = reshape(view(w, offset.+(1:prod(shape))), shape)
  wx = slice(0, (input, hidden*n))
  wh = slice(length(wx), (hidden, hidden*n))
  bias = view(w, length(wx)+length(wh) .+ (1:hidden*n))
  (wx, wh), bias
end

mutable struct RNNDesc{T}
  mode::cudnnRNNMode_t
  input::Int
  hidden::Int
  params::CuVector{T}
  weights::NTuple{2,CuMatrix{T}}
  bias::CuVector{T}
  ptr::Ptr{Nothing}
end

Base.unsafe_convert(::Type{Ptr{Nothing}}, d::RNNDesc) = d.ptr

function rnnParamSize(T, r, input)
  size = Csize_t[0]
  cudnnGetRNNParamsSize(handle(), r, TensorDesc(T, (1,input,1)), size, cudnnDataType(T))
  return Int(size[])÷sizeof(T)
end

ngates(mode) = [1, 1, 4, 3][mode+1]
ngates(r::RNNDesc) = ngates(r.mode)

function RNNDesc{T}(mode::cudnnRNNMode_t, input::Int, hidden::Int; layers = 1) where T
  d = [C_NULL]
  cudnnCreateRNNDescriptor(d)

  dropoutDesc = DropoutDesc(0)
  inputMode = CUDNN_LINEAR_INPUT
  direction = CUDNN_UNIDIRECTIONAL
  algo = CUDNN_RNN_ALGO_STANDARD
  cudnnSetRNNDescriptor_v6(handle(),d[],hidden,layers,dropoutDesc,cudnnRNNInputMode_t(inputMode),cudnnDirectionMode_t(direction),mode,cudnnRNNAlgo_t(algo),cudnnDataType(T))

  w =CuArrays.zeros(T, rnnParamSize(T, d[], input))
  # TODO: avoid reserve allocation here
  rd = RNNDesc{T}(mode, input, hidden, w, params(w, input, hidden, ngates(mode))..., d[])
  finalizer(rd) do x
    cudnnDestroyRNNDescriptor(x)
  end
  return rd
end

function setweights!(d::RNNDesc, Wi, Wh, b)
  transpose!(d.weights[1], Wi)
  transpose!(d.weights[2], Wh)
  transpose!(d.bias, b)
  return
end

function cudnnGetRNNTrainingReserveSize(r::RNNDesc, seqlen, xdesc)
  size = Csize_t[0]
  cudnnGetRNNTrainingReserveSize(handle(), r, seqlen, xdesc, size)
  return Int(size[])
end

function cudnnRNNForward(rnn::RNNDesc{T}, seqlen, xd, x, hd, h, cd, c, wd, w, yd, y, hod,
                         ho, cod, co, reserve=nothing) where T
  @workspace size=@argout(
      cudnnGetRNNWorkspaceSize(handle(), rnn, seqlen, xd,
                               out(Ref{Csize_t}()))
    )[] workspace->begin
      if reserve == nothing
        cudnnRNNForwardInference(handle(), rnn, seqlen, xd, x, hd, h, cd, c, wd, w, yd, y,
                                hod, ho, cod, co, workspace, sizeof(workspace))
      else
        cudnnRNNForwardTraining(handle(), rnn, seqlen, xd, x, hd, h, cd, c, wd, w, yd, y,
                                hod, ho, cod, co, workspace, sizeof(workspace),
                                reserve, sizeof(reserve))
      end
    end
end

xDesc(x) = [TensorDesc(eltype(x), (1, size(x, 1), size(x, 2)))]

hDesc(h::Nothing) = C_NULL, CU_NULL
hDesc(x::Integer) = (@assert x == 0; hDesc(nothing))
function hDesc(h::CuArray)
  TensorDesc(eltype(h), (size(h, 1), size(h, 2), 1)), h
end

# TODO: can we just manipulate strides here?
# TODO: should use repmat, but this isn't implemented.
hBatch(x::AbstractVector, h::CuVector) = h
hBatch(x::AbstractMatrix, h::CuVector) = h .*CuArrays.ones(1, size(x, 2))
hBatch(x::AbstractMatrix, h::CuMatrix) = h .*CuArrays.ones(1, size(h,2) == 1 ? size(x,2) : 1)

function forward(rnn::RNNDesc{T}, x::CuArray{T}, h_::CuArray{T}, c_ = nothing, train = Val{false}) where T
  h = hBatch(x, h_)
  c = c_ == nothing ? nothing : hBatch(x, c_)
  @assert size(x, 1) == rnn.input
  @assert size(h, 1) == rnn.hidden
  @assert size(x, 2) == size(h, 2)
  seqLength = 1
  xdesc = xDesc(x)
  y = x isa AbstractVector ? similar(x, rnn.hidden) : similar(x, rnn.hidden, size(x, 2))
  ho = similar(h)
  ydesc = xDesc(y)
  reserve = train == Val{true} ?
    CuVector{UInt8}(undef, cudnnGetRNNTrainingReserveSize(rnn, seqLength, xdesc)) :
    nothing
  co = c == nothing ? c : similar(c)
  cudnnRNNForward(rnn, seqLength,
                  xdesc, x,
                  hDesc(h)...,
                  hDesc(c)...,
                  FilterDesc(T, (1, 1, length(rnn.params))), rnn.params,
                  ydesc, y,
                  hDesc(ho)...,
                  hDesc(co)...,
                  reserve)
  result = c == nothing ? (y, ho) : (y, ho, co)
  return train == Val{true} ? (reserve, result) : result
end

forwardTrain(rnn::RNNDesc{T}, x::CuArray{T}, h::CuArray{T}, c = nothing) where T =
  forward(rnn, x, h, c, Val{true})

function cudnnRNNBackwardData(rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc,
                              dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc,
                              dx, dhxDesc, dhx, dcxDesc, dcx, reserve)
  @workspace size=@argout(
      cudnnGetRNNWorkspaceSize(handle(), rnnDesc, seqLength, dxDesc,
                               out(Ref{Csize_t}()))
    )[] workspace->begin
      cudnnRNNBackwardData(handle(), rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc,
                           dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc,
                           dx, dhxDesc, dhx, dcxDesc, dcx, workspace, sizeof(workspace),
                           reserve, sizeof(reserve))
    end
end

function backwardData(rnn::RNNDesc{T}, y, dy_, dho, dco, h, c, reserve) where T
  # Same as above, any more efficient way?
  dy = dy_ isa Integer ? zero(y) : dy_
  yd = xDesc(y)
  dx = y isa AbstractVector ? similar(dy, rnn.input) : similar(dy, rnn.input, size(dy, 2))
  dh = similar(h)
  dc = c == nothing ? nothing : similar(c)
  cudnnRNNBackwardData(rnn, 1, yd, y, yd, dy, hDesc(dho)..., hDesc(dco)...,
                       FilterDesc(T, (1, 1, length(rnn.params))), rnn.params, hDesc(h)...,
                       hDesc(c)..., xDesc(dx), dx, hDesc(dh)..., hDesc(dc)..., reserve)
  return c == nothing ? (dx, dh) : (dx, dh, dc)
end

backwardData(rnn, y, dy, dho, hx, reserve) =
  backwardData(rnn, y, dy, dho, nothing, hx, nothing, reserve)

function cudnnRNNBackwardWeights(rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc,
                                 y, dwDesc, dw, reserve)
  @workspace size=@argout(
      cudnnGetRNNWorkspaceSize(handle(), rnnDesc, seqLength, xDesc,
                               out(Ref{Csize_t}()))
    )[] workspace->begin
      cudnnRNNBackwardWeights(handle(), rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc,
                              y, workspace, sizeof(workspace), dwDesc, dw,
                              reserve, sizeof(reserve))
    end
end

function backwardWeights(rnn::RNNDesc{T}, x, h, y, reserve) where T
  dw = zero(rnn.params)
  cudnnRNNBackwardWeights(rnn, 1, xDesc(x), x, hDesc(h)..., xDesc(y), y,
                          FilterDesc(T, (1, 1, length(dw))), dw, reserve)
  return params(dw, rnn.input, rnn.hidden, ngates(rnn))
end

function pullback(rnn::RNNDesc{T}, x::CuArray{T}, h::CuArray{T}) where T <: Union{Float32,Float64}
  reserve, (y, ho) = CUDNN.forwardTrain(rnn, x, h)
  return (y, ho), function (dy, dho)
    h_ = CUDNN.hBatch(x, h)
    dx, dh = CUDNN.backwardData(rnn, y, dy, dho, h_, reserve)
    (dWi, dWh), db = CUDNN.backwardWeights(rnn, x, h_, y, reserve)
    return (x = dx, h = dh, Wi = dWi, Wh = dWh, b = db)
  end
end

function pullback(rnn::RNNDesc{T}, x::CuArray{T}, h::CuArray{T}, c::CuArray{T}) where T <: Union{Float32,Float64}
  reserve, (y, ho, co) = CUDNN.forwardTrain(rnn, x, h, c)
  return (y, ho, co), function (dy, dho, dco)
    h_ = CUDNN.hBatch(x, h)
    c_ = CUDNN.hBatch(x, c)
    dx, dh, dc = CUDNN.backwardData(rnn, y, dy, dho, dco, h_, c_, reserve)
    (dWi, dWh), db = CUDNN.backwardWeights(rnn, x, h_, y, reserve)
    return (x = dx, h = dh, c = dc, Wi = dWi, Wh = dWh, b = db)
  end
end
