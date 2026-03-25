using Test

using CUDACore
using cuBLAS
using LinearAlgebra
using Random
using Adapt: adapt

# compare CPU vs GPU results (simplified version of GPUArrays.TestSuite.compare)
function testf(f, xs...; kwargs...)
    cpu_in = map(x -> x isa Base.RefValue ? x[] : deepcopy(x), xs)
    gpu_in = map(x -> x isa Base.RefValue ? x[] : adapt(CuArray, x), xs)
    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    _compare(cpu_out, gpu_out; kwargs...)
end
function _compare(a::AbstractArray, b::AbstractArray; kwargs...)
    return ≈(collect(a), collect(b); kwargs...)
end
function _compare(a::Number, b::Number; kwargs...)
    return ≈(a, b isa AbstractArray ? collect(b)[] : b; kwargs...)
end
function _compare(as::Tuple, bs::Tuple; kwargs...)
    all(zip(as, bs)) do (a, b)
        _compare(a, b; kwargs...)
    end
end
_compare(a, b; kwargs...) = a == b
