import KernelAbstractions
import KernelAbstractions as KA
import KernelInterface as KI

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

# sparse is tested by cuSPARSE; the others run kernels on KA's POCL-based CPU back-end
ka_skip_tests = Set{String}(["sparse", "CPU synchronization", "fallback test: callable types"])
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

    # XXX: KernelAbstractions' launcher converts the arguments twice: to compile the
    #      kernel (`KI.argconvert`), and again when launching it (`KI.launch`)
    @test_broken counter[] == 1
    @test Array(output) == collect(1:257)

    counter[] = 0
    broadcast_input = CUDA.fill(1f0, 257)
    broadcast_output = similar(broadcast_input)
    broadcast_output .= broadcast_input .+ KAConversionHost(2f0, counter)
    synchronize()

    @test_broken counter[] == 1
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

KA.@kernel function store_global_linear!(A)
    I = KA.@index(Global, Linear)
    @inbounds A[I] = I
end

KA.@kernel function store_last_index!(A)
    I = KA.@index(Global, Linear)
    if I == prod(KA.@ndrange())
        @inbounds A[1] = I
        @inbounds A[2] = KA.@index(Global, Cartesian)[2]
    end
end

@testset "launch configuration" begin
    backend = CUDABackend()
    function select(kernel, ndrange, workgroupsize=nothing)
        ndrange, workgroupsize, iterspace, _ = KA.launch_config(kernel, ndrange, workgroupsize)
        KA.select_launch(kernel, workgroupsize, iterspace)
    end

    # kernels are launched on an N-d grid, computing indices in 32 bits
    kernel = store_global_linear!(backend)
    @test select(kernel, (64, 32, 16)) === KA.NDLaunch{Int32}()
    @test select(kernel, (4, 4, 4, 4)) === KA.LinearLaunch{Int32}()
    @test select(kernel, (8, 100_000)) === KA.LinearLaunch{Int32}()

    # which doesn't need divisions to compute the index of a dynamic N-d range
    A = CUDA.zeros(Int, 64, 32, 16)
    ptx = sprint(io -> CUDA.@device_code_ptx io=io kernel(A; ndrange=size(A)))
    @test !occursin("div.", ptx)
    @test !occursin("rem.", ptx)
    @test Array(A) == LinearIndices(A)

    # tuning for more blocks (the testsuite above uses the default)
    Testsuite.launch_testsuite(()->CUDABackend(; prefer_blocks=true), CuArray)

    # iteration spaces that don't fit 32 bits use 64-bit indices
    kernel = store_last_index!(backend)
    A = CUDA.zeros(Int, 2)
    for (dims, launch) in (((2^16 + 1, 2^15), KA.NDLaunch{Int}()),
                           ((2^11 + 1, 2^10, 2^10, 1), KA.LinearLaunch{Int}()))
        @test select(kernel, dims) === launch
        kernel(A; ndrange=dims)
        @test Array(A) == [prod(dims), dims[2]]
    end
end

function ki_store_index!(A)
    i = KI.get_global_id().x
    if i <= length(A)
        @inbounds A[i] = i
    end
    return
end

@testset "compiler options and tuning" begin
    # a static workgroup size bounds the number of threads per block
    A = CUDA.zeros(Int, 1024)
    kernel = store_global_linear!(CUDABackend(), 256)
    ptx = sprint(io -> CUDA.@device_code_ptx io=io kernel(A; ndrange=length(A)))
    @test occursin(".maxntid 256", ptx)
    @test Array(A) == 1:1024

    # tuning receives the number of work-items; prefer_blocks launches more, smaller blocks
    kernel = KI.@launch CUDABackend() launch=false ki_store_index!(A)
    threads = KI.launch_configuration(kernel; nitems=1024).workgroupsize
    @test threads <= 1024
    kernel = KI.@launch CUDABackend(; prefer_blocks=true) launch=false ki_store_index!(A)
    fewer = KI.launch_configuration(kernel; nitems=1024).workgroupsize
    @test fewer < threads
    # ... but not for a bound on the workgroup size alone
    @test KI.launch_configuration(kernel; max_work_group_size=1024).workgroupsize == threads
    KI.@launch CUDABackend(; prefer_blocks=true) ndrange=length(A) ki_store_index!(A)
    @test Array(A) == 1:1024
end
