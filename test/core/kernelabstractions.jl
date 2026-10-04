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
# host arrays are converted to device arrays with 32-bit indices
ka_device_array = CuDeviceArray{T,N,A,Int32} where {T,N,A}
Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, ka_device_array;
                    skip_tests=ka_skip_tests)
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDA, ka_device_array;
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

KA.@kernel function add_denormals!(a, b)
    i = KA.@index(Global)
    @inbounds a[i] += b[i]
end

@testset "fastmath" begin
    # Fast math flushes Float32 subnormals (add.ftz.f32). Switch back to IEEE mode too,
    # to check that compilation caches the option.
    for fastmath in (false, true, false), workgroupsize in (nothing, 32)
        a = CUDA.zeros(Float32, 2)
        b = CuArray([nextfloat(0.0f0), -nextfloat(0.0f0)])
        kernel = add_denormals!(CUDABackend(; fastmath))
        kernel(a, b; ndrange=2, workgroupsize)
        @test Array(a) == (fastmath ? zeros(Float32, 2) : Array(b))
    end
end

@testset "index types" begin
    # launches with arrays that don't fit 32-bit indices take a different path
    KA.@kernel function size_kernel!(out, a)
        @inbounds out[1] = size(a, 2)
    end
    out = CuArray([0])
    a = CuArray{Float32}(undef, 2, 3)
    size_kernel!(CUDABackend())(out, a; ndrange=1)
    @test Array(out) == [3]
    GC.@preserve a begin
        big = unsafe_wrap(CuArray, pointer(a), (2, 2^30))   # never accessed
        size_kernel!(CUDABackend())(out, big; ndrange=1)
        @test Array(out) == [2^30]
    end
end
