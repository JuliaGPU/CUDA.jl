using Test

using CUDACore
using cuBLAS
using LinearAlgebra
using Random
using Adapt: adapt

# compare CPU vs GPU results (simplified version of GPUArrays.TestSuite.compare).
# @nospecialize on the closure and argument tuple: this forwarder has no per-type
# computation, and each unique (f, xs...) call site would otherwise produce its own
# specialization — the GPUArrays.TestSuite.compare equivalent was the #1 testsuite
# compile hotspot before it was @nospecialize'd.
function testf(@nospecialize(f), @nospecialize(xs...); kwargs...)
    cpu_in = map(x -> x isa Base.RefValue ? x[] : deepcopy(x), xs)
    gpu_in = map(x -> x isa Base.RefValue ? x[] : adapt(CuArray, x), xs)
    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    _compare(cpu_out, gpu_out; kwargs...)
end
function _compare(@nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); kwargs...)
    return ≈(collect(a), collect(b); kwargs...)
end
function _compare(a::Number, b::Number; kwargs...)
    return ≈(a, b isa AbstractArray ? collect(b)[] : b; kwargs...)
end
function _compare(@nospecialize(as::Tuple), @nospecialize(bs::Tuple); kwargs...)
    all(zip(as, bs)) do (a, b)
        _compare(a, b; kwargs...)
    end
end
_compare(@nospecialize(a), @nospecialize(b); kwargs...) = a == b
