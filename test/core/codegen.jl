@testset "LLVM IR" begin

@testset "JuliaLang/julia#21121" begin
    @test @filecheck CUDA.code_llvm(Tuple{}) do
        @check_not "inttoptr"
        weight_matrix = CuStaticSharedArray(Float32, (16, 16))
        sync_threads()
        weight_matrix[1, 16] *= 2
        sync_threads()
    end
end

@testset "CUDA.jl#553" begin
    @test @filecheck CUDA.code_llvm(Tuple{Ptr{Float32}}) do ptr
        @check_not "@__nv_fmaf"
        unsafe_store!(ptr, CUDA.fma(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
        return
    end
end

@testset "fma uses LLVM intrinsic" begin
    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"), (Float16, "f16"))
        @test @filecheck CUDA.code_llvm(Tuple{Ptr{T}}) do ptr
            @check "llvm.fma.$suffix"
            @check_not "__nv_fma"
            unsafe_store!(ptr, fma(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
            return
        end
    end
end

@testset "muladd uses LLVM intrinsic" begin
    # `Base.muladd` emits `fmul contract + fadd contract` upstream, which the
    # backend usually fuses to `fma.rn`. On GPU the fusion is unreliable under
    # vectorization (JuliaGPU/CUDA.jl#3149), so the override routes through
    # `llvm.fmuladd.fXX` directly.
    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"), (Float16, "f16"))
        @test @filecheck CUDA.code_llvm(Tuple{Ptr{T}}) do ptr
            @check "llvm.fmuladd.$suffix"
            unsafe_store!(ptr, muladd(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
            return
        end
    end
end

@testset "assume" begin
    @test @filecheck CUDA.code_llvm(Tuple{Int}) do i
        @check "@gpu_report_exception"
        cld(42, i)
    end

    @test @filecheck CUDA.code_llvm(Tuple{Int}) do i
        @check_not "gpu_report_exception"
        CUDA.assume(i > 0)
        cld(42, i)
    end
end

@testset "stripping invariant.load" begin
    function kernel(ptr, x)
        i = CUDACore.threadIdx_x()
        @inbounds ptr[] = x[i]
        return
    end

    arr = CuArray(zeros(Float64))

    @cuda kernel(arr, (1., 2., ))
    @test Array(arr)[] == 1.
end

@testset "stripping const TBAA" begin
    # this one is particularly nasty because it occurs in a nested function

    _a = rand(Int, 2, 1)
    b = ((1,9999),(1,9999))

    out = CuArray(zeros(Int, 2,1))
    a = Tuple(_a)

    function kernel(out, a, b)
        i = threadIdx().x
        blockIdx().x
        @inbounds out[i,1] = a[i] + b[i][1]
        return
    end

    @cuda threads=2 kernel(out, a, b)
    @test Array(out) == (_a .+ 1)
end

@testset "ptxas-compatible control flow" begin
    @noinline function throw_some()
        throw(42)
        return
    end

    @inbounds function kernel(input, output, n)
        i = threadIdx().x

        temp = CuStaticSharedArray(Int, 1)
        if i == 1
            1 <= n || throw_some()
            temp[1] = input
        end
        sync_threads()

        1 <= n || throw_some()
        unsafe_store!(output, temp[1], i)

        return
    end

    function gpu(input)
        output = CuArray(zeros(eltype(input), 2))
        ptr = pointer(output)
        ptr = reinterpret(Ptr{eltype(input)}, ptr)

        @cuda threads=2 kernel(input, ptr, 99)

        return Array(output)
    end

    function cpu(input)
        output = zeros(eltype(input), 2)

        for j in 1:2
            @inbounds output[j] = input
        end

        return output
    end

    input = rand(1:100)
    @test cpu(input) == gpu(input)
end

end

############################################################################################

@testset "PTX" begin

@testset "always_inline" begin
    # without `always_inline`, the helper survives as a separate `.func`;
    # with it set, the helper is inlined and no `.func julia_f_expensive`
    # declaration remains. The closure-form lambdas below recreate the
    # `f_expensive` helper at each test site, so each parent has its own
    # call edge to verify the kwarg sticks.
    f_expensive(x) = (Base.Cartesian.@nexprs 30 i -> x = sin(x)+i; x)
    for always_inline in (false, true)
        @test @filecheck CUDA.code_ptx(Tuple{Float64}; always_inline) do x
            @check     cond=!always_inline "{{\\.func .*julia_f_expensive}}"
            @check_not cond=always_inline  "{{\\.func .*julia_f_expensive}}"
            f_expensive(x)
            return
        end
    end
end

@testset "local memory stores due to byval" begin
    # JuliaGPU/GPUCompiler.jl#92
    @test @filecheck CUDA.code_ptx(NTuple{2,CuDeviceArray{Float32,1,AS.Global}}) do y1, y2
        @check_not ".local"
        y = threadIdx().x == 1 ? y1 : y2
        @inbounds y[] = 0
        return
    end

    # dynamically-indexed aggregate arguments should load directly from parameter space
    # instead of being copied to local memory first
    @test @filecheck CUDA.code_ptx(Tuple{CuDeviceArray{Float32,1,AS.Global},
                                         NTuple{32,Float32}, Int}) do out, t, i
        @check_not ".local"
        @inbounds out[1] = t[i]
        return
    end
end

@testset "header rewrite (.target/.version bump)" begin
    # When the LLVM back-end can't reach the device cap (e.g., a device newer
    # than what NVPTX_LLVM_Backend_jll supports), `_compiler_config` produces a
    # split config and `mcgen` rewrites `.target`/`.version` in the emitted asm.
    # `.attribute(.unified)` is target-gated on sm_90+ across CUDA 12.0+ —
    # picked here as a stable cross-toolkit feature gate that exercises the
    # rewrite without requiring such hardware in CI.
    asm_pre = """
    .version 8.0
    .target sm_75
    .address_size 64

    .global .attribute(.unified(19, 95)) .f32 f;
    """

    function run_ptxas(src::String, gpu::String)
        ptx = tempname() * ".ptx"; write(ptx, src)
        out = tempname() * ".cubin"
        opts = ["--compile-only", "--gpu-name", gpu, "--output-file", out, ptx]
        proc = run(pipeline(ignorestatus(`$(CUDACore.CUDA_Compiler.ptxas()) $opts`);
                            stdout=devnull, stderr=devnull); wait=true)
        rm(ptx; force=true); rm(out; force=true)
        proc
    end

    @test !success(run_ptxas(asm_pre, "sm_75"))

    asm_post = CUDACore.rewrite_ptx_header(asm_pre, v"8.0", sm"90")
    @test occursin(".target sm_90", asm_post)

    @test success(run_ptxas(asm_post, "sm_90"))

    # Architecture-specific feature set appends an `a` suffix to the .target directive (and the same
    # string is what `compile()` passes to --gpu-name, since ptxas requires exact match for `a`-mode).
    asm_arch = CUDACore.rewrite_ptx_header(asm_pre, v"8.0", sm"90a")
    @test occursin(".target sm_90a", asm_arch)
    @test success(run_ptxas(asm_arch, "sm_90a"))

    # Family-specific appends `f`. Requires PTX 8.8+ at the `.target` line.
    asm_family = CUDACore.rewrite_ptx_header(asm_pre, v"8.8", sm"100f")
    @test occursin(".target sm_100f", asm_family)
    @test success(run_ptxas(asm_family, "sm_100f"))
end

@testset "SMVersion and sm\"...\" macro" begin
    @test sm"90"   == SMVersion(9, 0, :baseline)
    @test sm"90a"  == SMVersion(9, 0, :arch)
    @test sm"100f" == SMVersion(10, 0, :family)
    # printing roundtrips via the macro form
    @test sprint(show, sm"103a") == "sm\"103a\""
    @test sprint(show, sm"100")  == "sm\"100\""
    # cpu_name reflects feature_set
    @test CUDACore.cpu_name(sm"90")   == "sm_90"
    @test CUDACore.cpu_name(sm"90a")  == "sm_90a"
    @test CUDACore.cpu_name(sm"100f") == "sm_100f"
    # base_version drops the suffix back to a comparable VersionNumber
    @test CUDACore.base_version(sm"103a") == v"10.3"
    # constructor rejects bogus feature_set
    @test_throws ErrorException SMVersion(9, 0, :bogus)
    # macro rejects malformed strings
    @test_throws ErrorException parse(SMVersion, "10.3a")    # dotted form (NVIDIA uses dotless)
    @test_throws ErrorException parse(SMVersion, "100x")     # unknown suffix
    @test_throws ErrorException parse(SMVersion, "1")        # only one digit (need at least major + minor)
    @test_throws ErrorException parse(SMVersion, "")         # empty

    # `SMVersion(x)` as the universal normalizer:
    @test SMVersion(sm"103a")          === sm"103a"                        # identity
    @test SMVersion(v"10.3")           == SMVersion(10, 3, :baseline)      # VersionNumber → baseline
    @test SMVersion("103a")            == sm"103a"                         # bare string
    @test SMVersion("sm_103a")         == sm"103a"                         # accepts NVIDIA prefix
    # the macro is just a parse-time call to the constructor
    @test sm"103a"                     == SMVersion("103a")
end

end

############################################################################################

@testset "host reference patching" begin
    runtime_refs = @eval module $(gensym())
        using ..CUDACore
        const host_reference_type_tag = CUDACore.GPUCompiler.Runtime.type_tag
        host_reference_kernel(out) = (out[1] = host_reference_type_tag(Val(:float32)); return)
    end

    out = CUDA.zeros(UInt, 1)
    @cuda threads=1 runtime_refs.host_reference_kernel(out)
    expected = UInt(unsafe_load(cglobal(:jl_float32_type, Ptr{UInt})))
    @test Array(out)[] == expected

    value_refs = @eval module $(gensym())
        const host_reference_symbol = Symbol("value#global")
        host_reference_symbol_kernel(out, name) =
            (out[1] = UInt(name === host_reference_symbol); return)
    end

    out = CUDA.zeros(UInt, 1)
    @cuda threads=1 value_refs.host_reference_symbol_kernel(out, Symbol("value#global"))
    @test Array(out)[] == 1

    interior_refs = @eval module $(gensym())
        @noinline produce(cond::Bool, value::Int32) = cond ? value : 1.5
        function interior_reference_kernel(out, cond::Bool, value::Int32)
            x = produce(cond, value)
            out[1] = UInt(x isa Float64)
            return
        end
    end

    out = CUDA.zeros(UInt, 1)
    kernel = @cuda launch=false interior_refs.interior_reference_kernel(out, false, Int32(1))
    kernel(out, false, Int32(1); threads=1)
    @test Array(out)[] == 1
    @test any(root -> root === Float64, kernel.roots)
end

############################################################################################

@testset "SASS" begin

@testset "basic reflection" begin
    valid_kernel() = return
    invalid_kernel() = 1

    if can_use_cupti() && !(v"2024.2.0" <= CUPTI.library_version()) # NVIDIA bug #4667039
        @test CUDA.code_sass(devnull, valid_kernel, Tuple{}) == nothing
        @test_throws CUDA.KernelError CUDA.code_sass(devnull, invalid_kernel, Tuple{})
    end
end

@testset "function name mangling" begin
    @eval @noinline $(Symbol("dummy_^"))(x) = x

    @eval kernel_341(ptr) = (@inbounds unsafe_store!(ptr, $(Symbol("dummy_^"))(unsafe_load(ptr))); nothing)

    if can_use_cupti() && !(v"2024.2.0" <= CUPTI.library_version()) # NVIDIA bug #4667039
        CUDA.code_sass(devnull, kernel_341, Tuple{Ptr{Int}})
    end
end

@testset "device runtime" begin
    kernel() = (CUDACore.cudaGetLastError(); return)

    if can_use_cupti() && !(v"2024.2.0" <= CUPTI.library_version()) # NVIDIA bug #4667039
        CUDA.code_sass(devnull, kernel, Tuple{})
    end
end

end
