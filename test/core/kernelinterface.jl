import KernelInterface
import KernelInterface as KI
using CUDACore

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(CUDABackend(), CuArray)

@testset "copyto!" begin
    backend = CUDABackend()

    # bits unions store their type tags separately
    host = Union{Missing, Int32}[1, missing, 3]
    dev = CuArray{Union{Missing, Int32}}(undef, 3)
    @test KI.copyto!(backend, dev, host) === dev
    back = Vector{Union{Missing, Int32}}(undef, 3)
    KI.copyto!(backend, back, dev)
    KI.synchronize(backend)
    @test isequal(back, host)

    # host to host
    a = zeros(Float32, 4)
    @test KI.copyto!(backend, a, ones(Float32, 4)) === a
    @test a == ones(Float32, 4)

    # contiguous views of host arrays (those of a `CuArray` are `CuArray`s)
    dev = CUDA.zeros(Float32, 4)
    host = Float32[1, 2, 3, 4, 5, 6]
    @test KI.copyto!(backend, dev, view(host, 2:5)) === dev
    KI.synchronize(backend)
    @test Array(dev) == [2, 3, 4, 5]
    KI.copyto!(backend, view(host, 1:4), CUDA.ones(Float32, 4))
    KI.synchronize(backend)
    @test host == [1, 1, 1, 1, 5, 6]

    # only contiguous arrays
    @test_throws ArgumentError KI.copyto!(backend, view(CUDA.zeros(Float32, 8), 1:2:8), CUDA.ones(Float32, 4))
end

function ki_fill!(A)
    i = KI.get_global_id().x
    if i <= length(A)
        @inbounds A[i] = i
    end
    return
end

@testset "launch keywords" begin
    A = CUDA.zeros(Int, 4)
    kernel = KI.@launch CUDABackend() launch=false ki_fill!(A)

    # CUDA's launch options are passed on
    kernel(A; ndrange=4, stream=stream())
    @test Array(A) == 1:4

    # but not ones that would override the launch geometry
    @test_throws ArgumentError kernel(A; ndrange=4, threads=8)
    @test_throws ArgumentError kernel(A; ndrange=4, blocks=2)
end
