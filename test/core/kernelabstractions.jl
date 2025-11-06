import KernelAbstractions
import KernelAbstractions as KA

struct KAConversionHost{T}
    value::T
    counter::Base.RefValue{Int}
end

struct KAConversionDevice{T}
    value::T
end

Base.broadcastable(arg::KAConversionHost) = Ref(arg)
Base.:+(x::Float32, arg::KAConversionDevice{Float32}) = x + arg.value

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, arg::KAConversionHost)
    arg.counter[] += 1
    KAConversionDevice(Adapt.adapt(to, arg.value))
end

KA.@kernel function copy_converted!(output, arg)
    index = KA.@index(Global)
    @inbounds output[index] = arg.wrapper.value[index]
end

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

ka_skip_tests = Set{String}(["sparse"])
Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, CuDeviceArray; skip_tests=Set([
    "CPU synchronization",
    "fallback test: callable types",]))
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDA, CuDeviceArray;
                                 skip_tests=ka_skip_tests)
end

@testset "KA.functional" begin
    @test KA.functional(CUDABackend()) == CUDA.functional()
end

@testset "argument conversion" begin
    backend = CUDABackend()
    kernel = copy_converted!(backend)
    input = CuArray(collect(1:257))
    output = similar(input)
    counter = Ref(0)
    arg = (wrapper=KAConversionHost(input, counter),)

    kernel(output, arg; ndrange=length(output))
    synchronize()

    counter[] = 0
    kernel(output, arg; ndrange=length(output))
    synchronize()

    @test counter[] == 1
    @test Array(output) == collect(1:257)

    counter[] = 0
    broadcast_input = CUDA.fill(1f0, 257)
    broadcast_output = similar(broadcast_input)
    broadcast_output .= broadcast_input .+ KAConversionHost(2f0, counter)
    synchronize()

    @test counter[] == 1
    @test Array(broadcast_output) == fill(3f0, 257)
end
