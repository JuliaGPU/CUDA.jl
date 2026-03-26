using Test

using CUDACore
using cuBLAS
using cuSPARSE
using cuSOLVER

using LinearAlgebra
using Adapt: adapt

# compare CPU vs GPU results
function testf(f, xs...; kwargs...)
    cpu_in = map(x -> x isa Base.RefValue ? x[] : deepcopy(x), xs)
    gpu_in = map(x -> x isa Base.RefValue ? x[] : adapt(CuArray, x), xs)
    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    _compare(cpu_out, gpu_out; kwargs...)
end
_compare(a::AbstractArray, b::AbstractArray; kwargs...) = ≈(collect(a), collect(b); kwargs...)
_compare(a::Number, b::Number; kwargs...) = ≈(a, b isa AbstractArray ? collect(b)[] : b; kwargs...)
_compare(as::Tuple, bs::Tuple; kwargs...) = all(zip(as, bs)) do (a, b); _compare(a, b; kwargs...); end
_compare(a, b; kwargs...) = a == b
