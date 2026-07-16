dummy() = return

struct CountedHost{T}
    value::T
    counter::Base.RefValue{Int}
end

struct CountedDevice{T}
    value::T
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, arg::CountedHost)
    arg.counter[] += 1
    CountedDevice(Adapt.adapt(to, arg.value))
end

function copy_counted(arg, output)
    output[] = arg.value[]
    return
end

@testset "@cuda" begin

@test_throws UndefVarError @cuda undefined()
@test_throws MethodError @cuda dummy(1)


@testset "launch configuration" begin
    @cuda dummy()

    threads = 1
    @cuda threads dummy()
    @cuda threads=1 dummy()
    @cuda threads=(1,1) dummy()
    @cuda threads=(1,1,1) dummy()

    blocks = 1
    @cuda blocks dummy()
    @cuda blocks=1 dummy()
    @cuda blocks=(1,1) dummy()
    @cuda blocks=(1,1,1) dummy()

    clustersize = 1
    @cuda clustersize dummy()
    @cuda clustersize=1 dummy()
    @cuda clustersize=(1,1) dummy()
    @cuda clustersize=(1,1,1) dummy()
end


@testset "launch=false" begin
    k = @cuda launch=false dummy()
    k()
    k(; threads=1)

    CUDA.version(k)
    CUDA.memory(k)
    CUDA.registers(k)
    CUDA.maxthreads(k)
end


@testset "kernel invocation" begin
    input = CuArray([11])
    output = CuArray([0])

    counter = Ref(0)
    arg = CountedHost(input, counter)
    @cuda copy_counted(arg, output)
    @test counter[] == 1
    @test Array(output) == [11]

    counter[] = 0
    invocation = CUDA.prepare(copy_counted, arg, output)
    @test invocation isa CUDA.KernelInvocation
    @test invocation[1] === arg
    kernel = CUDA.compile(invocation)
    @test kernel isa CUDACore.HostKernel
    @test CUDACore.launch_configuration(kernel.fun).threads > 0
    CUDA.launch(kernel, invocation)
    CUDA.launch(kernel, invocation)
    @test counter[] == 1
    @test Array(output) == [11]
    rebound = CUDA.prepare(kernel, arg, output)
    @test rebound.backend isa CUDA.LLVMBackend
    @test counter[] == 2

    counter[] = 0
    kernel = @cuda launch=false copy_counted(arg, output)
    @test counter[] == 1
    kernel(arg, output)
    @test counter[] == 2

    replacement_counter = Ref(0)
    replacement = CountedHost(CuArray([12]), replacement_counter)
    replacement_invocation = Base.setindex(invocation, replacement, 1)
    @test invocation[1] === arg
    @test replacement_invocation[1] === replacement
    @test replacement_counter[] == 1
    GC.gc(true)
    CUDA.launch(kernel, replacement_invocation)
    @test replacement_counter[] == 1
    @test Array(output) == [12]

    ephemeral_counter = Ref(0)
    function escaping_invocation()
        CUDA.prepare(copy_counted,
                     CountedHost(CuArray([14]), ephemeral_counter), output)
    end
    escaped = escaping_invocation()
    GC.gc(true)
    escaped_kernel = CUDA.compile(escaped)
    CUDA.launch(escaped_kernel, escaped)
    @test ephemeral_counter[] == 1
    @test Array(output) == [14]

    empty_invocation = CUDA.prepare(dummy)
    CUDA.launch(CUDA.compile(empty_invocation), empty_invocation)

    replace_int(inv, value) = Base.setindex(inv, value, 1)
    int_invocation = CUDA.prepare(identity, Int32(1))
    @test @inferred(replace_int(int_invocation, Int32(2))) isa CUDA.KernelInvocation
    replace_int(int_invocation, Int32(2))
    @test @allocated(replace_int(int_invocation, Int32(2))) == 0
    runtime_index = Ref(1)[]
    @test Base.setindex(int_invocation, Int64(3), runtime_index)[1] == 3

    function write_scalar(value, output)
        output[] = value
        return
    end
    scalar_output = CuArray(Int32[0])
    scalar_invocation = CUDA.prepare(write_scalar, Int32(1), scalar_output)
    scalar_kernel = CUDA.compile(scalar_invocation)
    widened_invocation = Base.setindex(scalar_invocation, Int64(2), 1)
    CUDA.launch(scalar_kernel, widened_invocation)
    @test Array(scalar_output) == Int32[2]
    @test_throws InexactError CUDA.launch(
        scalar_kernel, Base.setindex(scalar_invocation, typemax(Int64), 1))

    scalar_kernel(Int64(3), scalar_output)
    @test Array(scalar_output) == Int32[3]

    float_input = CuArray(Float32[1])
    function copy_first(input, output)
        output[] = input[]
        return
    end
    array_invocation = CUDA.prepare(copy_first, input, output)
    array_kernel = CUDA.compile(array_invocation)
    @test_throws MethodError CUDA.launch(
        array_kernel, Base.setindex(array_invocation, float_input, 1))
end


@testset "compilation params" begin
    llvm22 = CUDACore.llvm_compat(v"22")
    @test CUDACore.default_ptx_versions(llvm22, CUDACore.ptxas_compat(v"12.9")) ==
          (v"8.8", v"8.8")
    @test CUDACore.default_ptx_versions(llvm22, CUDACore.ptxas_compat(v"13.3")) ==
          (v"9.0", v"9.3")

    # Don't assume the compatibility sets are contiguous.
    llvm_support = (ptx=Set([v"8.7", v"9.0"]),)
    ptxas_support = (ptx=Set([v"8.7", v"8.8"]),)
    @test CUDACore.default_ptx_versions(llvm_support, ptxas_support) ==
          (v"8.7", v"8.8")

    @cuda dummy()

    @test_throws "Number of threads per block exceeds kernel limit" begin
        @cuda threads=2 maxthreads=1 dummy()
    end
    @cuda threads=2 dummy()

    # Older targets may be known to LLVM/PTX, but not to any CUDA version we support.
    @test_throws "requires compute capability sm_50" @cuda launch=false arch=sm"10" dummy()
    @test_throws "requires compute capability sm_50" @cuda launch=false arch=sm"20" dummy()
    # there isn't any capability other than the device's that's guaruanteed to work
    dev_cap = capability(device())
    dev_sm = SMVersion(dev_cap.major, dev_cap.minor)
    @cuda launch=false arch=dev_sm dummy()
    # `arch=` also accepts a plain `VersionNumber` -- treated as baseline. Equivalent
    # to constructing the SMVersion directly.
    @cuda launch=false arch=dev_cap dummy()
    # but we should be able to see it in the generated PTX code
    @test @filecheck CUDA.code_ptx((); arch=sm"50") do
        @check ".target sm_50"
        dummy()
    end
    @test @filecheck CUDA.code_ptx((); arch=v"5.0") do
        @check ".target sm_50"
        dummy()
    end

    # explicit `ptx=` is taken as an exact request (codegen-test affordance), so the
    # `.version` line should match what was asked for, independently of what LLVM and
    # ptxas would natively pick.
    @test @filecheck CUDA.code_ptx((); ptx=v"8.0") do
        @check ".version 8.0"
        dummy()
    end
    @test_throws "requires PTX ISA 8.0" @cuda launch=false ptx=v"7.8" dummy()

    # explicit `ptx=` is validated against BOTH LLVM and ptxas (not just LLVM as it
    # used to be); a clearly out-of-range value must error at config time.
    @test_throws "not supported" @cuda launch=false ptx=v"99.0" dummy()

    # feature_set is selected by the suffix on the sm"..." string; the suffix should
    # surface in the .target directive in the PTX output. The cuda-side `.target` is
    # the variant regardless of LLVM support -- the mcgen rewrite stamps it in even
    # when LLVM clamped to baseline for codegen.
    sm_a = SMVersion(dev_cap.major, dev_cap.minor, :arch)
    sm_f = SMVersion(dev_cap.major, dev_cap.minor, :family)

    if dev_cap >= v"9.0"
        @test @filecheck CUDA.code_ptx((); arch=sm_a) do
            @check ".target $(CUDACore.cpu_name(sm_a))"
            dummy()
        end
        # arch-specific cubin should also actually launch on the matching device
        @cuda arch=sm_a dummy()
    end
    if dev_cap >= v"10.0"
        @test @filecheck CUDA.code_ptx((); arch=sm_f) do
            @check ".target $(CUDACore.cpu_name(sm_f))"
            dummy()
        end
        @cuda arch=sm_f dummy()
    end

    # `cap=` is the deprecated alias for `arch=`; check the depwarn fires while
    # the path still produces the right PTX.
    @test_deprecated sprint(io->CUDA.code_ptx(io, dummy, (); cap=sm"50"))

    # With no explicit `arch=`, we default to architecture-specific code paths on CC >=9.0
    # since we know the exact device. The cuda-side `.target` is the variant regardless of
    # LLVM support (the mcgen rewrite stamps it in); only the LLVM-emitted code differs.
    if dev_cap >= v"9.0"
        @test @filecheck CUDA.code_ptx(()) do
            @check ".target $(CUDACore.cpu_name(sm_a))"
            dummy()
        end
    end

    # `target_feature_set()` reads back the feature set the *LLVM-emitted* code was built
    # for (not the cuda-side .target): when LLVM doesn't natively support the exact variant,
    # we fall back to baseline LLVM, so the global reflects baseline. The if-chain folds at
    # codegen time, so the launched kernel writes a single constant.
    function read_feature_set!(out)
        @inbounds out[1] = if target_feature_set() === :arch
            UInt32(2)
        elseif target_feature_set() === :family
            UInt32(1)
        else
            UInt32(0)
        end
        cc = compute_capability()
        ptx = ptx_isa_version()
        @inbounds out[2] = cc.major
        @inbounds out[3] = cc.minor
        @inbounds out[4] = ptx.major
        @inbounds out[5] = ptx.minor
        return
    end
    out = CUDA.fill(typemax(UInt32), 5)
    @cuda threads=1 read_feature_set!(out)
    # arch features come through `target_feature_set()` only when the back-end LLVM
    # natively supports the variant; otherwise we fell back to baseline and the
    # global reflects that.
    arch_in_llvm = sm_a in CUDACore.llvm_compat().sm
    expected = dev_cap >= v"9.0" && arch_in_llvm ? UInt32(2) : UInt32(0)
    target = CUDACore.compiler_config(device()).target
    @test Array(out) == UInt32[expected, target.cap.major, target.cap.minor,
                               target.ptx.major, target.ptx.minor]
end


@testset "launch failure: opt-in shmem + thread overrun" begin
    # A non-SMEM launch failure on a kernel that opted into dynamic SMEM beyond
    # the non-opt-in ceiling must report the real cause, not a spurious "exceeds
    # device limit" SMEM message. Only meaningful when the device exposes an
    # opt-in cap above the non-opt-in ceiling (Volta+).
    nonoptin = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    optin    = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    if optin > nonoptin
        k = @cuda launch=false maxthreads=1 dummy()
        attributes(k.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = optin
        @test_throws "exceeds kernel limit" k(; threads=2, shmem=optin)
    end
end


@testset "launch failure: register pressure" begin
    # Force ptxas to keep N live values across a barrier. The stores into
    # `out` are the observable side effect that prevents the optimizer from
    # eliminating the loads as dead code.
    function reg_kernel(out, inp, ::Val{N}) where N
        i = threadIdx().x
        v = ntuple(j -> inp[i + (j-1)*N], Val(N))
        sync_threads()
        ntuple(j -> (out[i + (j-1)*N] = v[j]; nothing), Val(N))
        return
    end
    N = 256
    hw_cap = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    workspace = CuArray{Float32}(undef, hw_cap + N^2)
    k       = @cuda launch=false reg_kernel(workspace, workspace, Val(N))
    nregs   = CUDA.registers(k)
    reglim  = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    threads = fld(reglim, nregs) + 1
    @test_throws "Block register count exceeds device limit" begin
        @assert threads > CUDA.maxthreads(k)
        k(workspace, workspace; threads=threads)
    end
end


@testset "inference" begin
    foo() = @cuda dummy()
    @inferred foo()

    # with arguments, we call cudaconvert
    kernel(a) = return
    bar(a) = @cuda kernel(a)
    @inferred bar(CuArray([1]))

    function reassigned_launch_kwarg()
        threads = 1
        @cuda threads=threads dummy()
        threads = 2
        return threads
    end
    lowered = only(code_lowered(reassigned_launch_kwarg, Tuple{}))
    @test !occursin("Core.Box", sprint(show, lowered))
end


@testset "shared memory" begin
    @cuda shmem=1 dummy()
end


@testset "backend dispatch" begin
    # instance form: explicit LLVMBackend
    @cuda backend=CUDACore.LLVMBackend() dummy()
    k = @cuda launch=false backend=CUDACore.LLVMBackend() dummy()
    @test k isa CUDACore.HostKernel
    k()

    # instance form via the re-exported CUDA name
    @cuda backend=CUDA.LLVMBackend() dummy()

    # module form: CUDA / CUDACore both define DefaultBackend()
    @cuda backend=CUDA dummy()
    @cuda backend=CUDACore dummy()

    # custom backend stub: assert prepare/compile/launch are called
    # and that unknown kwargs are forwarded
    @eval module BackendStub
        using CUDA
        const compile_calls = Ref(0)
        const convert_calls = Ref(0)
        const launch_calls = Ref(0)
        const last_kwargs = Ref{Any}(nothing)
        const last_launch_kwargs = Ref{Any}(nothing)

        struct Backend <: CUDACore.AbstractBackend end
        DefaultBackend() = Backend()

        struct Kernel{F,TT} <: CUDACore.AbstractKernel{F,TT}
            f::F
        end

        CUDACore.backend(::Kernel) = Backend()
        CUDACore.prepare(::Backend, x) = (convert_calls[] += 1; x)
        function CUDACore.compile(::Backend, f::F, tt::Type{<:Tuple};
                                  kwargs...) where {F}
            compile_calls[] += 1
            last_kwargs[] = (; kwargs...)
            Kernel{F,tt}(f)
        end
        function (kernel::Kernel)(args...; kwargs...)
            launch_calls[] += 1
            last_launch_kwargs[] = (; kwargs...)
            nothing
        end
    end

    BackendStub.compile_calls[] = 0
    BackendStub.convert_calls[] = 0
    BackendStub.last_kwargs[] = nothing
    @cuda backend=BackendStub.Backend() dummy()
    @test BackendStub.compile_calls[] == 1
    # f + 0 args, all routed through prepare
    @test BackendStub.convert_calls[] == 1
    @test BackendStub.launch_calls[] == 1

    # module-as-backend resolution for the custom backend
    @cuda backend=BackendStub dummy()
    @test BackendStub.compile_calls[] == 2

    # other_kwargs forwarding to compile
    @cuda backend=BackendStub.Backend() opt_level=2 dummy()
    @test haskey(BackendStub.last_kwargs[], :opt_level)
    @test BackendStub.last_kwargs[][:opt_level] == 2

    @cuda backend=BackendStub.Backend() opt_level=3 threads=16 dummy()
    @test BackendStub.last_kwargs[] == (opt_level=3,)
    @test BackendStub.last_launch_kwargs[] == (threads=16,)

    BackendStub.compile_calls[] = 0
    BackendStub.convert_calls[] = 0
    BackendStub.launch_calls[] = 0
    invocation = CUDA.prepare(dummy, 1, 2; backend=BackendStub)
    @test invocation isa CUDA.KernelInvocation
    @test invocation[1] == 1
    kernel = CUDA.compile(invocation; opt_level=3)
    @test kernel isa BackendStub.Kernel
    CUDA.launch(kernel, invocation; threads=4)
    invocation = Base.setindex(invocation, 3, 1)
    CUDA.launch(kernel, invocation; threads=8)
    drifted = Base.setindex(invocation, 4.0, 2)
    @test drifted[2] == 4.0
    @test BackendStub.compile_calls[] == 1
    @test BackendStub.convert_calls[] == 5 # function, two arguments, and two replacements
    @test BackendStub.launch_calls[] == 2
    @test BackendStub.last_kwargs[] == (opt_level=3,)
    @test BackendStub.last_launch_kwargs[] == (threads=8,)

    # dynamic + backend kwargs are rejected (errors at macro-expansion time,
    # so wrap in @macroexpand to defer)
    @test_throws "does not support backend-specific" begin
        @macroexpand @cuda dynamic=true opt_level=2 dummy()
    end
end


@testset "streams" begin
    s = CuStream()
    @cuda stream=s dummy()
end

@testset "clusters" begin
    cooperative = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1
    if cooperative
        @cuda cooperative=true dummy()
    end

    if CUDA.capability(device()) >= v"9.0"
        @cuda threads=64 blocks=2 clustersize=2 dummy()
        if cooperative
            @cuda cooperative=true threads=64 blocks=2 clustersize=2 dummy()
        end
    else
        @test_throws CuError @cuda threads=64 blocks=2 clustersize=2 dummy()
    end
end

@testset "external kernels" begin
    @eval module KernelModule
        export external_dummy
        external_dummy() = return
    end
    import .KernelModule
    @cuda KernelModule.external_dummy()
    @eval begin
        using .KernelModule
        @cuda external_dummy()
    end

    @eval module WrapperModule
        using CUDA
        @eval dummy() = return
        wrapper() = @cuda dummy()
    end
    WrapperModule.wrapper()
end


@testset "calling device function" begin
    @noinline child(i) = sink(i)
    function parent()
        child(1)
        return
    end

    @cuda parent()
end


@testset "varargs" begin
    function kernel(args...)
        @cuprint(args[2])
        return
    end

    _, out = @grab_output begin
        @cuda kernel(1, 2, 3)
    end
    @test out == "2"
end

end


############################################################################################

@testset "argument passing" begin

dims = (16, 16)
len = prod(dims)

@testset "manually allocated" begin
    function kernel(input, output)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x

        val = input[i]
        output[i] = val

        return
    end

    input = round.(rand(Float32, dims) * 100)
    output = similar(input)

    input_dev = CuArray(input)
    output_dev = CuArray(output)

    @cuda threads=len kernel(input_dev, output_dev)
    @test input ≈ Array(output_dev)
end


@testset "scalar through single-value array" begin
    function kernel(a, x)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        max = gridDim().x * blockDim().x
        if i == max
            _val = a[i]
            x[] = _val
        end
        return
    end

    arr = round.(rand(Float32, dims) * 100)
    val = [0f0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda threads=len kernel(arr_dev, val_dev)
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "scalar through single-value array, using device function" begin
    @noinline child(a, i) = a[i]
    function parent(a, x)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        max = gridDim().x * blockDim().x
        if i == max
            _val = child(a, i)
            x[] = _val
        end
        return
    end

    arr = round.(rand(Float32, dims) * 100)
    val = [0f0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda threads=len parent(arr_dev, val_dev)
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "tuples" begin
    # issue #7: tuples not passed by pointer

    function kernel(keeps, out)
        if keeps[1]
            out[] = 1
        else
            out[] = 2
        end
        return
    end

    keeps = (true,)
    d_out = CuArray(zeros(Int))

    @cuda kernel(keeps, d_out)
    @test Array(d_out)[] == 1
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)
    c = similar(a)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)

    @eval struct ExecGhost end

    function kernel(ghost, a, b, c)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        c[i] = a[i] + b[i]
        return
    end
    @cuda threads=len kernel(ExecGhost(), d_a, d_b, d_c)
    @test a+b == Array(d_c)


    # bug: ghost type function parameters confused aggregate type rewriting

    function kernel(ghost, out, aggregate)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        out[i] = aggregate[1]
        return
    end
    @cuda threads=len kernel(ExecGhost(), d_c, (42,))

    @test all(val->val==42, Array(d_c))
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    function kernel(ptr, b)
        ptr[] = imag(b)
        return
    end

    arr = CuArray(zeros(Float32))
    x = ComplexF32(2,2)

    @cuda kernel(arr, x)
    @test Array(arr)[] == imag(x)
end


@testset "automatic recompilation" begin
    arr = CuArray(zeros(Int))

    function kernel(ptr)
        ptr[] = 1
        return
    end

    @cuda kernel(arr)
    @test Array(arr)[] == 1

    function kernel(ptr)
        ptr[] = 2
        return
    end

    @cuda kernel(arr)
    @test Array(arr)[] == 2
end


@testset "automatic recompilation (bis)" begin
    arr = CuArray(zeros(Int))

    @eval doit(ptr) = ptr[] = 1

    function kernel(ptr)
        doit(ptr)
        return
    end

    @cuda kernel(arr)
    @test Array(arr)[] == 1

    @eval doit(ptr) = ptr[] = 2

    @cuda kernel(arr)
    @test Array(arr)[] == 2
end


@testset "non-isbits arguments" begin
    function kernel1(T, i)
        sink(i)
        return
    end
    @cuda kernel1(Int, 1)

    function kernel2(T, i)
        sink(unsafe_trunc(T,i))
        return
    end
    @cuda kernel2(Int, 1.)
end


@testset "splatting" begin
    function kernel(out, a, b)
        out[] = a+b
        return
    end

    out = [0]
    out_dev = CuArray(out)

    @cuda kernel(out_dev, 1, 2)
    @test Array(out_dev)[1] == 3

    all_splat = (out_dev, 3, 4)
    @cuda kernel(all_splat...)
    @test Array(out_dev)[1] == 7

    partial_splat = (5, 6)
    @cuda kernel(out_dev, partial_splat...)
    @test Array(out_dev)[1] == 11
end

@testset "object invoke" begin
    # this mimics what is generated by closure conversion

    @eval struct KernelObject{T} <: Function
        val::T
    end

    function (self::KernelObject)(a)
        a[] = self.val
        return
    end

    function outer(a, val)
       inner = KernelObject(val)
       @cuda inner(a)
    end

    a = [1.]
    a_dev = CuArray(a)

    outer(a_dev, 2.)

    @test Array(a_dev) ≈ [2.]
end

@testset "closures" begin
    function outer(a_dev, val)
       function inner(a)
            # captures `val`
            a[] = val
            return
       end
       @cuda inner(a_dev)
    end

    a = [1.]
    a_dev = CuArray(a)

    outer(a_dev, 2.)

    @test Array(a_dev) ≈ [2.]
end

@testset "closure as arguments" begin
    function kernel(closure)
        closure()
        return
    end
    function outer(a_dev, val)
        f() = a_dev[] = val
        @cuda kernel(f)
    end

    a = [1.]
    a_dev = CuArray(a)

    outer(a_dev, 2.)

    @test Array(a_dev) ≈ [2.]
end

@testset "conversions" begin
    @eval struct Host   end
    @eval struct Device end

    Adapt.adapt_storage(::CUDA.KernelAdaptor, a::Host) = Device()

    Base.convert(::Type{Int}, ::Host)   = 1
    Base.convert(::Type{Int}, ::Device) = 2

    out = [0]

    # convert arguments
    out_dev = CuArray(out)
    let arg = Host()
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg)
            return
        end
        @cuda kernel(arg, out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # convert captured variables
    out_dev = CuArray(out)
    let arg = Host()
        @test Array(out_dev) ≈ [0]
        function kernel(out)
            out[] = convert(Int, arg)
            return
        end
        @cuda kernel(out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # convert tuples
    out_dev = CuArray(out)
    let arg = (Host(),)
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg[1])
            return
        end
        @cuda kernel(arg, out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # convert named tuples
    out_dev = CuArray(out)
    let arg = (a=Host(),)
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg.a)
            return
        end
        @cuda kernel(arg, out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # don't convert structs
    out_dev = CuArray(out)
    @eval struct Nested
        a::Host
    end
    let arg = Nested(Host())
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg.a)
            return
        end
        @cuda kernel(arg, out_dev)
        @test Array(out_dev) ≈ [1]
    end
end

@testset "argument count" begin
    val = [0]
    val_dev = CuArray(val)
    for i in (1, 10, 20, 34)
        variables = ('a':'z'..., 'A':'Z'...)
        params = [Symbol(variables[j]) for j in 1:i]
        # generate a kernel
        body = quote
            function kernel(arr, $(params...))
                arr[] = $(Expr(:call, :+, params...))
                return
            end
        end
        eval(body)
        args = [j for j in 1:i]
        call = Expr(:call, :kernel, val_dev, args...)
        cudacall = :(@cuda $call)
        eval(cudacall)
        @test Array(val_dev)[1] == sum(args)
    end
end

@testset "keyword arguments" begin
    @eval inner_kwargf(foobar;foo=1, bar=2) = nothing

    @cuda (()->inner_kwargf(42;foo=1,bar=2))()

    @cuda (()->inner_kwargf(42))()

    @cuda (()->inner_kwargf(42;foo=1))()

    @cuda (()->inner_kwargf(42;bar=2))()

    @cuda (()->inner_kwargf(42;bar=2,foo=1))()
end

@testset "captured values" begin
    function f(capture::T) where {T}
        function kernel(ptr)
            ptr[] = capture
            return
        end

        arr = CuArray(zeros(T))
        @cuda kernel(arr)

        return Array(arr)[1]
    end

    using Test
    @test f(1) == 1
    @test f(2) == 2
end

@testset "parameter space" begin
    kernel(x) = nothing
    @test_throws "Kernel invocation uses too much parameter memory" @cuda kernel(ntuple(_->UInt64(1), 2^13))
end

@testset "argument layout" begin
    # the back-end aligns 128-bit integers to 16 bytes, but Julia only started doing so in
    # 1.12, so aggregates with (U)Int128 fields lay out differently on older hosts. such
    # types are rejected there (host==device on 1.12+, so everything is accepted).
    @eval struct Int128Wrapper; x::Int64; y::Int128; end
    @eval struct FloatWrapper;  x::Int64; y::Float64; end   # control: no 128-bit integers
    host_ok = Base.datatype_alignment(Int128) == 16         # true on Julia 1.12+

    # -- the compatibility walk must look through type parameters (e.g. device-array
    #    element types), not just fields. this part is host-independent. --
    reaches_i128(T) = CUDACore.layout_reaches(S -> S === Int128 || S === UInt128, T)
    @test  reaches_i128(Int128)
    @test  reaches_i128(Int128Wrapper)                            # via a field
    @test  reaches_i128(Tuple{Int64,Int128Wrapper})              # via a tuple element
    @test  reaches_i128(Ptr{Int128Wrapper})                      # via a pointer's pointee
    @test  reaches_i128(CUDACore.CuDeviceArray{Int128Wrapper,1,1}) # via an element type
    @test !reaches_i128(Float64)
    @test !reaches_i128(FloatWrapper)
    @test !reaches_i128(CUDACore.CuDeviceArray{Float64,1,1})

    @test CUDACore.device_layout(Int128) == (16, 16)
    @test CUDACore.device_layout(FloatWrapper) == (16, 8)
    @test CUDACore.device_layout(Int128Wrapper) === (host_ok ? (32, 16) : :mismatch)
    @test CUDACore.device_compatible_layout(Int128Wrapper) == host_ok
    @test CUDACore.device_compatible_layout(CUDACore.CuDeviceArray{Int128Wrapper,1,1}) == host_ok
    @test CUDACore.device_compatible_layout(CUDACore.CuDeviceArray{Float64,1,1})

    # -- end-to-end: rejected on <1.12, compiled and correct on 1.12+ --

    # plain 128-bit integers occupy their own parameter slot / array element, and are fine
    # regardless of how the host aligns them
    setval(out, v) = (@inbounds out[1] = v; return)
    let out = CuArray{Int128}(undef, 1)
        @cuda setval(out, Int128(2)^100 + 7)
        @test Array(out)[1] == Int128(2)^100 + 7
    end

    # aggregate with a 128-bit field, passed as a kernel argument: read its field back so
    # the layout check (not an unrelated type error) is what fails on incompatible hosts
    gety(out, w) = (@inbounds out[1] = w.y; return)
    if host_ok
        out = CuArray{Int128}(undef, 1)
        @cuda gety(out, Int128Wrapper(42, Int128(2)^100 + 7))
        @test Array(out)[1] == Int128(2)^100 + 7
    else
        out = CuArray{Int128}(undef, 1)
        @test_throws "references 128-bit integer fields" @cuda gety(out, Int128Wrapper(42, Int128(2)^100 + 7))
    end

    # aggregate with a 128-bit field, reached as a device-array element (memory traffic;
    # this is only seen through a pointer, so it used to slip past the argument check)
    function readfields(out, A)
        i = threadIdx().x
        @inbounds begin
            out[2i-1] = A[i].x
            out[2i]   = A[i].y % Int64
        end
        return
    end
    wrappers = [Int128Wrapper(1, 2), Int128Wrapper(3, 4)]
    if host_ok
        dA = CuArray(wrappers); out = CuArray{Int64}(undef, 4)
        @cuda threads=2 readfields(out, dA)
        @test Array(out) == [1, 2, 3, 4]
    else
        dA = CuArray(wrappers); out = CuArray{Int64}(undef, 4)
        @test_throws "references 128-bit integer fields" @cuda threads=2 readfields(out, dA)
    end

    # control: an aggregate without 128-bit integers is always accepted
    let out = CuArray{Float64}(undef, 1)
        @cuda gety(out, FloatWrapper(1, 3.5))
        @test Array(out)[1] == 3.5
    end
end

    @testset "symbols" begin
        function pass_symbol(x, name)
            i = name == :var ? 1 : 2
            x[i] = true
            return nothing
        end
        x = CuArray([false, false])
        @cuda pass_symbol(x, :var)
        @test Array(x) == [true, false]
        @cuda pass_symbol(x, :not_var)
        @test Array(x) == [true, true]

        function pass_mixed(x, name, value)
            i = name == :var ? 1 : 2
            x[i] = value
            return nothing
        end
        y = CUDA.zeros(Int, 2)
        @cuda pass_mixed(y, :var, 42)
        @cuda pass_mixed(y, :not_var, 7)
        @test Array(y) == [42, 7]
    end

end

############################################################################################

@testset "shmem divergence bug" begin

@testset "trap" begin
    function trap()
        ccall("llvm.trap", llvmcall, Cvoid, ())
    end

    function kernel(input::Int32, output::Core.LLVMPtr{Int32}, yes::Bool=true)
        i = threadIdx().x

        temp = CuStaticSharedArray(Cint, 1)
        if i == 1
            yes || trap()
            temp[1] = input
        end
        sync_threads()

        yes || trap()
        unsafe_store!(output, temp[1], i)

        return nothing
    end

    input = rand(Cint(1):Cint(100))
    N = 2

    let output = CuArray(zeros(Cint, N))
        # defaulting to `true` embeds this info in the PTX module,
        # allowing `ptxas` to emit validly-structured code.
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr)
        @test Array(output) == fill(input, N)
    end

    let output = CuArray(zeros(Cint, N))
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr, true)
        @test Array(output) == fill(input, N)
    end
end

@testset "unreachable" begin
    function unreachable()
        @cuprintln("go home ptxas you're drunk")
        Base.llvmcall("unreachable", Cvoid, Tuple{})
    end

    function kernel(input::Int32, output::Core.LLVMPtr{Int32}, yes::Bool=true)
        i = threadIdx().x

        temp = CuStaticSharedArray(Cint, 1)
        if i == 1
            yes || unreachable()
            temp[1] = input
        end
        sync_threads()

        yes || unreachable()
        unsafe_store!(output, temp[1], i)

        return nothing
    end

    input = rand(Cint(1):Cint(100))
    N = 2

    let output = CuArray(zeros(Cint, N))
        # defaulting to `true` embeds this info in the PTX module,
        # allowing `ptxas` to emit validly-structured code.
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr)
        @test Array(output) == fill(input, N)
    end

    let output = CuArray(zeros(Cint, N))
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr, true)
        @test Array(output) == fill(input, N)
    end
end

@testset "mapreduce (full)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuArray{T, N}; dims = :, init...) where {T, N}
        OT = Float32
        v0 =  0.0f0

        threads = 256
        out = CuArray{OT}(undef, (1,))
        @cuda threads reduce_kernel(f, op, v0, A, Val{threads}(), out)
        Array(out)[1]
    end

    function reduce_kernel(f, op, v0::T, A, ::Val{LMEM}, result) where {T, LMEM}
        tmp_local = CuStaticSharedArray(T, LMEM)
        global_index = threadIdx().x
        acc = v0

        # Loop sequentially over chunks of input vector
        while global_index <= length(A)
            element = f(A[global_index])
            acc = op(acc, element)
            global_index += blockDim().x
        end

        # Perform parallel reduction
        local_index = threadIdx().x - 1i32
        @inbounds tmp_local[local_index + 1i32] = acc
        sync_threads()

        offset = blockDim().x ÷ 2
        while offset > 0
            @inbounds if local_index < offset
                other = tmp_local[local_index + offset + 1i32]
                mine = tmp_local[local_index + 1i32]
                tmp_local[local_index + 1i32] = op(mine, other)
            end
            sync_threads()
            offset = offset ÷ 2
        end

        if local_index == 0
            result[blockIdx().x] = @inbounds tmp_local[1i32]
        end

        return
    end

    A = rand(Float32, 1000)
    dA = CuArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

@testset "mapreduce (full, complex)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuArray{T, N}; dims = :, init...) where {T, N}
        OT = Complex{Float32}
        v0 =  0.0f0+0im

        threads = 256
        out = CuArray{OT}(undef, (1,))
        @cuda threads reduce_kernel(f, op, v0, A, Val{threads}(), out)
        Array(out)[1]
    end

    function reduce_kernel(f, op, v0::T, A, ::Val{LMEM}, result) where {T, LMEM}
        tmp_local = CuStaticSharedArray(T, LMEM)
        global_index = threadIdx().x
        acc = v0

        # Loop sequentially over chunks of input vector
        while global_index <= length(A)
            element = f(A[global_index])
            acc = op(acc, element)
            global_index += blockDim().x
        end

        # Perform parallel reduction
        local_index = threadIdx().x - 1i32
        @inbounds tmp_local[local_index + 1] = acc
        sync_threads()

        offset = blockDim().x ÷ 2
        while offset > 0
            @inbounds if local_index < offset
                other = tmp_local[local_index + offset + 1]
                mine = tmp_local[local_index + 1]
                tmp_local[local_index + 1] = op(mine, other)
            end
            sync_threads()
            offset = offset ÷ 2
        end

        if local_index == 0
            result[blockIdx().x] = @inbounds tmp_local[1]
        end

        return
    end

    A = rand(Complex{Float32}, 1000)
    dA = CuArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

@testset "mapreduce (reduced)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuArray{T, N}) where {T, N}
        OT = Int
        v0 = 0

        out = CuArray{OT}(undef, (1,))
        @cuda threads=64 reduce_kernel(f, op, v0, A, out)
        Array(out)[1]
    end

    function reduce_kernel(f, op, v0::T, A, result) where {T}
        tmp_local = CuStaticSharedArray(T, 64)
        acc = v0

        # Loop sequentially over chunks of input vector
        i = threadIdx().x
        while i <= length(A)
            element = f(A[i])
            acc = op(acc, element)
            i += blockDim().x
        end

        # Perform parallel reduction
        @inbounds tmp_local[threadIdx().x] = acc
        sync_threads()

        offset = blockDim().x ÷ 2
        while offset > 0
            @inbounds if threadIdx().x <= offset
                other = tmp_local[(threadIdx().x - 1i32) + offset + 1]
                mine = tmp_local[threadIdx().x]
                tmp_local[threadIdx().x] = op(mine, other)
            end
            sync_threads()
            offset = offset ÷ 2
        end

        if threadIdx().x == 1
            result[blockIdx().x] = @inbounds tmp_local[1]
        end

        return
    end

    A = rand(1:10, 100)
    dA = CuArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

end

############################################################################################

@testset "dynamic parallelism" begin

@testset "basic usage" begin
    function hello()
        @cuprint("Hello, ")
        @cuda dynamic=true world()
        return
    end

    @eval function world()
        @cuprint("World!")
        return
    end

    _, out = @grab_output begin
        @cuda hello()
    end
    @test out == "Hello, World!"
end

@testset "anonymous functions" begin
    function hello()
        @cuprint("Hello, ")
        world = () -> (@cuprint("World!"); nothing)
        @cuda dynamic=true world()
        return
    end

    _, out = @grab_output begin
        @cuda hello()
    end
    @test out == "Hello, World!"
end

@testset "closures" begin
    function hello()
        x = 1
        @cuprint("Hello, ")
        world = () -> (@cuprint("World $(x)!"); nothing)
        @cuda dynamic=true world()
        return
    end

    _, out = @grab_output begin
        @cuda hello()
    end
    @test out == "Hello, World 1!"
end

@testset "argument passing" begin
    ## padding

    function kernel(a, b, c)
        @cuprint("$a $b $c")
        return
    end

    for args in ((Int16(1), Int32(2), Int64(3)),    # padding
                 (Int32(1), Int32(2), Int32(3)),    # no padding, equal size
                 (Int64(1), Int32(2), Int16(3)),    # no padding, inequal size
                 (Int16(1), Int64(2), Int32(3)))    # mixed
        _, out = @grab_output begin
            @cuda kernel(args...)
        end
        @test out == "1 2 3"
    end

    ## conversion

    function kernel(a)
        increment(a) = (a[1] += 1; nothing)

        a[1] = 1
        increment(a)
        @cuda dynamic=true increment(a)

        return
    end

    dA = CuArray{Int}(undef, (1,))
    @cuda kernel(dA)
    A = Array(dA)
    @test A == [3]
end

@testset "self-recursion" begin
    @eval function kernel(x::Bool)
        if x
            @cuprint("recurse ")
            @cuda dynamic=true kernel(false)
        else
            @cuprint("stop")
        end
       return
    end

    _, out = @grab_output begin
        @cuda kernel(true)
    end
    @test out == "recurse stop"
end

@testset "deep recursion" begin
    @eval function kernel_a(x::Bool)
        @cuprint("a ")
        @cuda dynamic=true kernel_b(x)
        return
    end

    @eval function kernel_b(x::Bool)
        @cuprint("b ")
        @cuda dynamic=true kernel_c(x)
        return
    end

    @eval function kernel_c(x::Bool)
        @cuprint("c ")
        if x
            @cuprint("recurse ")
            @cuda dynamic=true kernel_a(false)
        else
            @cuprint("stop")
        end
        return
    end

    _, out = @grab_output begin
        @cuda kernel_a(true)
    end
    @test out == "a b c recurse a b c stop"
end

@testset "streams" begin
    function hello()
        @cuprint("Hello, ")
        s = CuDeviceStream()
        @cuda dynamic=true stream=s world()
        CUDA.unsafe_destroy!(s)
        return
    end

    @eval function world()
        @cuprint("World!")
        return
    end

    _, out = @grab_output begin
        @cuda hello()
    end
    @test out == "Hello, World!"
end

@testset "parameter alignment" begin
    # foo is unused, but determines placement of bar
    function child(x, foo, bar)
        x[] = sum(bar)
        return
    end
    function parent(x, foo, bar)
        @cuda dynamic=true child(x, foo, bar)
        return
    end

    for (Foo, Bar) in [(Tuple{},NTuple{8,Int}), # JuliaGPU/CUDA.jl#263
                        (Tuple{Int32},Tuple{Int16}),
                        (Tuple{Int16},Tuple{Int32,Int8,Int16,Int64,Int16,Int16})]
        foo = (Any[T(i) for (i,T) in enumerate(Foo.parameters)]...,)
        bar = (Any[T(i) for (i,T) in enumerate(Bar.parameters)]...,)

        x, y = CUDA.zeros(Int, 1), CUDA.zeros(Int, 1)
        @cuda child(x, foo, bar)
        @cuda parent(y, foo, bar)
        @test sum(bar) == Array(x)[] == Array(y)[]
    end
end

@testset "many arguments" begin
    # JuliaGPU/CUDA.jl#401
    function dp_5arg_kernel(v1, v2, v3, v4, v5)
        return nothing
    end

    function dp_6arg_kernel(v1, v2, v3, v4, v5, v6)
        return nothing
    end

    function main_5arg_kernel()
        @cuda threads=1 dynamic=true dp_5arg_kernel(1, 1, 1, 1, 1)
        return nothing
    end

    function main_6arg_kernel()
        @cuda threads=1 dynamic=true dp_6arg_kernel(1, 1, 1, 1, 1, 1)
        return nothing
    end

    @cuda threads=1 dp_5arg_kernel(1, 1, 1, 1, 1)
    @cuda threads=1 dp_6arg_kernel(1, 1, 1, 1, 1, 1)
    @cuda threads=1 main_5arg_kernel()
    @cuda threads=1 main_6arg_kernel()
end

end

############################################################################################

@testset "contextual dispatch" begin

@test_throws ErrorException CUDA.saturate(1f0)  # CUDA.jl#60

@test testf(a->broadcast(x->x^1.5, a), rand(Float32, 1))    # CUDA.jl#71
@test testf(a->broadcast(x->1.0^x, a), rand(Int, 1))        # CUDA.jl#76
@test testf(a->broadcast(x->x^4, a), rand(Float32, 1))      # CUDA.jl#171

@test argmax(cu([true false; false true])) == CartesianIndex(1, 1)  # CUDA.jl#659

# CUDA.jl#42
@test testf([Complex(1f0,2f0)]) do a
    b = sincos.(a)
    s,c = first(collect(b))
    (real(s), imag(s), real(c), imag(c))
end

end

############################################################################################

@testset "launch seed does not perturb host RNG" begin

    dummy_kernel() = return

    Random.seed!(0xdeadbeef)
    a_before = rand(UInt64)
    @cuda threads=1 dummy_kernel()
    a_after  = rand(UInt64)

    Random.seed!(0xdeadbeef)
    b_before = rand(UInt64)
    b_after  = rand(UInt64)

    @test a_before == b_before
    @test a_after  == b_after

    k = @cuda launch=false dummy_kernel()
    seed1 = CUDACore.make_seed(k)
    seed2 = CUDACore.make_seed(k)
    @test seed1 != seed2

end

############################################################################################
