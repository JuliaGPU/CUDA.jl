import KernelAbstractions
import KernelAbstractions as KA

struct KAConversionHost{T}
    value::T
    counter::Base.RefValue{Int}
end

struct KAConversionDevice{T}
    value::T
end


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
Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, CuDeviceArray;
                    skip_tests=ka_skip_tests)
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
end
