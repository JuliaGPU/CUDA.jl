using Test

using cuFFT

using CUDACore

import FFTW

using AbstractFFTs
using LinearAlgebra
using Adapt: adapt

# compare CPU vs GPU results. @nospecialize: see note in cublas/test/setup.jl.
function testf(@nospecialize(f), @nospecialize(xs...); kwargs...)
    cpu_in = map(x -> x isa Base.RefValue ? x[] : deepcopy(x), xs)
    gpu_in = map(x -> x isa Base.RefValue ? x[] : adapt(CuArray, x), xs)
    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    _compare(cpu_out, gpu_out; kwargs...)
end
_compare(@nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); kwargs...) = ≈(collect(a), collect(b); kwargs...)
_compare(a::Number, b::Number; kwargs...) = ≈(a, b isa AbstractArray ? collect(b)[] : b; kwargs...)
_compare(@nospecialize(a), @nospecialize(b); kwargs...) = a == b

# FFTW does not support Float16, so we roll our own

function AbstractFFTs.fft!(x::Array{Complex{Float16}}, dims...)
    y = Array{Complex{Float32}}(x)
    fft!(y, dims...)
    x .= y
end
function AbstractFFTs.bfft!(x::Array{Complex{Float16}}, dims...)
    y = Array{Complex{Float32}}(x)
    bfft!(y, dims...)
    x .= y
end
function AbstractFFTs.ifft!(x::Array{Complex{Float16}}, dims...)
    y = Array{Complex{Float32}}(x)
    ifft!(y, dims...)
    x .= y
end

AbstractFFTs.fft(x::Array{Complex{Float16}}, dims...) = Array{Complex{Float16}}(fft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.bfft(x::Array{Complex{Float16}}, dims...) = Array{Complex{Float16}}(bfft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.ifft(x::Array{Complex{Float16}}, dims...) = Array{Complex{Float16}}(ifft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.rfft(x::Array{Float16}, dims...) = Array{Complex{Float16}}(rfft(Array{Float32}(x), dims...))
AbstractFFTs.brfft(x::Array{Complex{Float16}}, dims...) = Array{Float16}(brfft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.irfft(x::Array{Complex{Float16}}, dims...) = Array{Float16}(irfft(Array{Complex{Float32}}(x), dims...))

struct WrappedFloat16Operator
    op
end
Base.:*(A::WrappedFloat16Operator, b::Array{Float16}) = Array{Float16}(A.op * Array{Float32}(b))
Base.:*(A::WrappedFloat16Operator, b::Array{Complex{Float16}}) = Array{Complex{Float16}}(A.op * Array{Complex{Float32}}(b))
function LinearAlgebra.mul!(C::Array{Float16}, A::WrappedFloat16Operator, B::Array{Float16}, α, β)
    C32 = Array{Float32}(C)
    B32 = Array{Float32}(B)
    mul!(C32, A.op, B32, α, β)
    C .= C32
end
function LinearAlgebra.mul!(C::Array{Complex{Float16}}, A::WrappedFloat16Operator, B::Array{Complex{Float16}}, α, β)
    C32 = Array{Complex{Float32}}(C)
    B32 = Array{Complex{Float32}}(B)
    mul!(C32, A.op, B32, α, β)
    C .= C32
end

function AbstractFFTs.plan_fft!(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_fft!(y, dims...))
end
function AbstractFFTs.plan_bfft!(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_bfft!(y, dims...))
end
function AbstractFFTs.plan_ifft!(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_ifft!(y, dims...))
end

function AbstractFFTs.plan_fft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_fft(y, dims...))
end
function AbstractFFTs.plan_bfft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_bfft(y, dims...))
end
function AbstractFFTs.plan_ifft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_ifft(y, dims...))
end
function AbstractFFTs.plan_rfft(x::Array{Float16}, dims...)
    y = similar(x, Float32)
    WrappedFloat16Operator(plan_rfft(y, dims...))
end
function AbstractFFTs.plan_irfft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_irfft(y, dims...))
end
function AbstractFFTs.plan_brfft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_brfft(y, dims...))
end

# notes:
#   plan_bfft does not need separate testing since it is used by plan_ifft

N1 = 8
N2 = 32
N3 = 64
# N4 is deliberately odd so that R2C external-batch strides are not
# multiples of sizeof(Complex{Float32}), exercising the alignment path.
N4 = 9
N5 = 2

rtol(::Type{Float16}) = 1e-2
rtol(::Type{Float32}) = 1e-5
rtol(::Type{Float64}) = 1e-12
rtol(::Type{I}) where {I<:Integer} = rtol(float(I))
atol(::Type{Float16}) = 1e-3
atol(::Type{Float32}) = 1e-8
atol(::Type{Float64}) = 1e-15
atol(::Type{I}) where {I<:Integer} = atol(float(I))
rtol(::Type{Complex{T}}) where {T} = rtol(T)
atol(::Type{Complex{T}}) where {T} = atol(T)
