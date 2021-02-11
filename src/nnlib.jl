using NNlib

# Activation functions
@cufunc softplus(x::Real) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

@cufunc logσ(x::Real) = -softplus(-x)

@cufunc function gelu(x::Real)
    p = oftype(x / 1, π)
    λ = oftype(x / 1, √(2 / p))
    α = oftype(x / 1, 0.044715)
    h = oftype(x / 1, 0.5)
    h * x * (one(x) + tanh(λ * (x + α * x^3)))
end

@cufunc lisht(x::Real) = x * tanh(x)

@cufunc logcosh(x::Real) = x + softplus(-2x) - log(oftype(x, 2))

@cufunc mish(x::Real) = x * tanh(softplus(x))

@cufunc tanhshrink(x::Real) = x - tanh(x)


# Batched matrix multiplication
# 1st argument is produced by NNlib.storage_type(A)
NNlib._batched_gemm!(::Type{<:CuArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C) =
     CUBLAS.gemm_strided_batched!(transA, transB, α, A, B, β, C)

Base.unsafe_convert(::Type{CuPtr{T}}, A::NNlib.BatchedAdjOrTrans{T}) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))


#
# Upsampling
#

# GPU based bilinear upsampling including its gradient
#
# Based on the Caffe2 implementation at:
# The code is a translation from the following files:
# - https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/caffe2/operators/upsample_op.cu
# - https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/caffe2/core/common_gpu.h
#
# Copyright (c) 2016-2021 Facebook Inc.
# Copyright (c) 2015 Google Inc.
# Copyright (c) 2015 Yangqing Jia
# Copyright 2019-2020 Kakao Brain
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials
#    provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America and
#    IDIAP Research Institute nor the names of its contributors may be used to endorse or
#    promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Forward and backward pass have been tested to produce the same output
# as pytorch with align_corners=True - it works modulo bit noise.

function upsample_bilinear_whcn_kernel!(n_elem, rheight, rwidth, x, y)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_w, in_h, channels, batchsize = size(x)
        out_w, out_h, _, _ = size(y)

        ow = index % out_w
        oh = index ÷ out_w

        real_index = rheight*oh
        ih0 = Base.floor(Int, real_index)
        offset = (ih0 < in_h-1) ? 1 : 0
        ih1 = ih0 + offset + 1
        h1lambda = real_index - ih0
        h0lambda = 1 - h1lambda
        ih0 += 1

        real_index = rwidth*ow
        iw0 = Base.floor(Int, real_index)
        offset = (iw0 < in_w-1) ? 1 : 0
        iw1 = iw0 + offset + 1
        w1lambda = real_index - iw0
        w0lambda = 1 - w1lambda
        iw0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = h0lambda * (w0lambda * x[iw0, ih0, c, n]  + # h0 * w0 * i00
                                  w1lambda * x[iw1, ih0, c, n]) + # h0 * w1 * i01
                      h1lambda * (w0lambda * x[iw0, ih1, c, n]  + # h1 * w0 * i10
                                  w1lambda * x[iw1, ih1, c, n])   # h1 * w1 * i11
                y[ow+1, oh+1, c, n] = val
            end
        end
    end
    return nothing
end

# Δ is the gradient backpropagated from downstream layers
function ∇upsample_bilinear_whcn_kernel!(n_elem, rheight, rwidth, Δ, dx)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_width, in_height, channels, batchsize = size(Δ)
        out_width, out_height, _, _ = size(dx)

        iw = index % in_width
        ih = index ÷ in_width

        # Compute Y axis lambdas
        real_index_h = rheight*ih
        oh0 = Base.floor(Int, real_index_h)
        offset = (oh0 < out_height-1) ? 1 : 0
        oh1 = oh0 + offset + 1
        h1lambda = real_index_h - oh0
        h0lambda = 1 - h1lambda
        oh0 += 1

        # # Compute X axis lambdas
        real_index_w = rwidth * iw
        ow0 = Base.floor(Int, real_index_w)
        offset = (ow0 < out_width - 1) ? 1 : 0
        ow1 = ow0 + offset + 1
        w1lambda = real_index_w - ow0
        w0lambda = 1 - w1lambda
        ow0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = Δ[iw+1, ih+1, c, n]
                @atomic dx[ow0, oh0, c, n] += h0lambda * w0lambda * val
                @atomic dx[ow1, oh0, c, n] += h0lambda * w1lambda * val
                @atomic dx[ow0, oh1, c, n] += h1lambda * w0lambda * val
                @atomic dx[ow1, oh1, c, n] += h1lambda * w1lambda * val
            end
        end
    end # if
    return nothing
end

function NNlib.upsample_bilinear_whcn!(y::CuArray{T,4}, x::CuArray{T,4}) where T
    w,h,c,n = size(x)
    out_w, out_h = (size(y,1), size(y,2))

    out_size = out_h*out_w
    rheight = T((h-1)/(out_h-1))
    rwidth  = T((w-1)/(out_w-1))

    kernel = @cuda launch=false upsample_bilinear_whcn_kernel!(out_size, rheight, rwidth, x, y)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = Base.min(out_size, config.threads)
    blocks = cld(out_size, threads)
    kernel(out_size, rheight, rwidth, x, y; threads=threads, blocks=blocks)
    return y
end

function NNlib.∇upsample_bilinear_whcn!(dx::CuArray{T,4}, Δ::CuArray{T,4}) where T
    w,h,c,n = Base.size(Δ)
    out_w, out_h = (size(dx, 1), size(dx, 2))
    in_size = h*w
    rheight = T((out_h-1)/(h-1)) # reversed compared to forward pass
    rwidth  = T((out_w-1)/(w-1))

    kernel = @cuda launch=false ∇upsample_bilinear_whcn_kernel!(in_size, rheight, rwidth, Δ, dx)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = Base.min(in_size, config.threads)
    blocks = cld(in_size, threads)
    kernel(in_size, rheight, rwidth, Δ, dx; threads=threads, blocks=blocks)
    return dx
end
