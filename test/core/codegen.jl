@testset "LLVM IR" begin

@testset "JuliaLang/julia#21121" begin
    function foobar()
        weight_matrix = CuStaticSharedArray(Float32, (16, 16))
        sync_threads()
        weight_matrix[1, 16] *= 2
        sync_threads()
    end

    ir = sprint(io->CUDA.code_llvm(io, foobar, Tuple{}))
    @test !occursin("inttoptr", ir)
end

@testset "CUDA.jl#553" begin
    function kernel(ptr)
       unsafe_store!(ptr, CUDA.fma(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
       return
    end

    ir = sprint(io->CUDA.code_llvm(io, kernel, Tuple{Ptr{Float32}}))
    @test !occursin("@__nv_fmaf", ir)
end

@testset "fma uses LLVM intrinsic" begin
    function fma_kernel(ptr)
        unsafe_store!(ptr, fma(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
        return
    end

    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"), (Float16, "f16"))
        ir = sprint(io->CUDA.code_llvm(io, fma_kernel, Tuple{Ptr{T}}))
        @test occursin("llvm.fma.$suffix", ir)
        @test !occursin("__nv_fma", ir)
    end
end

@testset "muladd uses LLVM intrinsic" begin
    function muladd_kernel(ptr)
        unsafe_store!(ptr, muladd(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
        return
    end

    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"), (Float16, "f16"))
        ir = sprint(io->CUDA.code_llvm(io, muladd_kernel, Tuple{Ptr{T}}))
        @test occursin("llvm.fmuladd.$suffix", ir)
    end
end

@testset "assume" begin
    foo(i) = cld(42, i)
    ir = sprint(io->CUDA.code_llvm(io, foo, Tuple{Int}))
    @test occursin("@gpu_report_exception", ir)


    bar(i) = (CUDA.assume(i > 0); cld(42, i))
    ir = sprint(io->CUDA.code_llvm(io, bar, Tuple{Int}))
    @test !occursin("gpu_report_exception", ir)
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
    function f_expensive(x)
        Base.Cartesian.@nexprs 30 i -> x = sin(x)+i
    end

    function g(x)
        f_expensive(x)
        return
    end
    function h(x)
        f_expensive(x)
        return
    end

    asm = sprint(io->CUDA.code_ptx(io, g, Tuple{Float64}))
    @test occursin(r"\.func .*julia_f_expensive", asm)

    asm = sprint(io->CUDA.code_ptx(io, g, Tuple{Float64}; always_inline=true))
    @test !occursin(r"\.func .*julia_f_expensive", asm)

    asm = sprint(io->CUDA.code_ptx(io, h, Tuple{Float64}; always_inline=true))
    @test !occursin(r"\.func .*julia_f_expensive", asm)

    asm = sprint(io->CUDA.code_ptx(io, h, Tuple{Float64}))
    @test occursin(r"\.func .*julia_f_expensive", asm)
end

@testset "local memory stores due to byval" begin
    # JuliaGPU/GPUCompiler.jl#92
    function kernel(y1, y2)
        y = threadIdx().x == 1 ? y1 : y2
        @inbounds y[] = 0
        return
    end

    asm = sprint(io->CUDA.code_ptx(io, kernel, NTuple{2,CuDeviceArray{Float32,1,AS.Global}}))
    @test !occursin(".local", asm)
end

@testset "fastmath" begin
    function div_kernel(x)
        i = threadIdx().x
        @fastmath @inbounds x[i] = 1 / x[i]
        return
    end

    asm = sprint(io->CUDA.code_ptx(io, div_kernel, Tuple{CuDeviceArray{Float32,1,AS.Global}}; fastmath=true))
    @test occursin("div.approx.ftz", asm)

    function sqrt_kernel(x)
        i = threadIdx().x
        @inbounds x[i] = sqrt(x[i])
        return
    end

    asm = sprint(io->CUDA.code_ptx(io, sqrt_kernel, Tuple{CuDeviceArray{Float32,1,AS.Global}}))
    @test occursin("sqrt.r", asm)

    asm = sprint(io->CUDA.code_ptx(io, sqrt_kernel, Tuple{CuDeviceArray{Float32,1,AS.Global}}; fastmath=true))
    @test occursin("sqrt.approx.ftz", asm)
end

@testset "fma/muladd emit fma.rn" begin
    # fma and muladd should both lower to fma.rn in PTX
    function fma_kernel(a, b, c)
        @inbounds a[] = fma(b[], c[], a[])
        return
    end
    function muladd_kernel(a, b, c)
        @inbounds a[] = muladd(b[], c[], a[])
        return
    end

    for T in (Float16, Float32, Float64)
        asm = sprint(io->CUDA.code_ptx(io, fma_kernel,
            NTuple{3,CuDeviceArray{T,1,AS.Global}}))
        @test occursin("fma.rn", asm)
        @test !occursin("__nv_fma", asm)

        asm = sprint(io->CUDA.code_ptx(io, muladd_kernel,
            NTuple{3,CuDeviceArray{T,1,AS.Global}}))
        @test occursin("fma.rn", asm)
    end
end

@testset "header rewrite (.target/.version bump)" begin
    # When LLVM's NVPTX backend can't reach the device cap (e.g. Julia 1.12 +
    # LLVM 18 on a Blackwell device), `_compiler_config` produces a split
    # config and `mcgen` rewrites `.target`/`.version` in the emitted asm.
    # `.attribute(.unified)` is target-gated on sm_90+ across CUDA 12.0+ —
    # picked here as a stable cross-toolkit feature gate that exercises the
    # rewrite without requiring Blackwell hardware in CI.
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

    asm_post = CUDACore.rewrite_ptx_header(asm_pre, v"8.0", sm"9.0")
    @test occursin(".target sm_90", asm_post)

    @test success(run_ptxas(asm_post, "sm_90"))

    # Architecture-specific feature set appends an `a` suffix to the .target directive (and the same
    # string is what `compile()` passes to --gpu-name, since ptxas requires exact match for `a`-mode).
    asm_arch = CUDACore.rewrite_ptx_header(asm_pre, v"8.0", sm"9.0a")
    @test occursin(".target sm_90a", asm_arch)
    @test success(run_ptxas(asm_arch, "sm_90a"))

    # Family-specific appends `f`. Requires PTX 8.8+ at the `.target` line.
    asm_family = CUDACore.rewrite_ptx_header(asm_pre, v"8.8", sm"10.0f")
    @test occursin(".target sm_100f", asm_family)
    @test success(run_ptxas(asm_family, "sm_100f"))
end

@testset "SMVersion and sm\"...\" macro" begin
    @test sm"9.0"   == SMVersion(9, 0, :baseline)
    @test sm"9.0a"  == SMVersion(9, 0, :arch)
    @test sm"10.0f" == SMVersion(10, 0, :family)
    # printing roundtrips via the macro form
    @test sprint(show, sm"10.3a") == "sm\"10.3a\""
    @test sprint(show, sm"10.0")  == "sm\"10.0\""
    # cpu_name reflects feature_set
    @test CUDACore.cpu_name(sm"9.0")   == "sm_90"
    @test CUDACore.cpu_name(sm"9.0a")  == "sm_90a"
    @test CUDACore.cpu_name(sm"10.0f") == "sm_100f"
    # base_version drops the suffix back to a comparable VersionNumber
    @test CUDACore.base_version(sm"10.3a") == v"10.3"
    # constructor rejects bogus feature_set
    @test_throws ErrorException SMVersion(9, 0, :bogus)
    # macro rejects malformed strings
    @test_throws ErrorException parse(SMVersion, "103a")     # missing dot
    @test_throws ErrorException parse(SMVersion, "10.3x")    # unknown suffix
    @test_throws ErrorException parse(SMVersion, "10")       # missing minor
end

@testset "CUDACompilerParams hash discriminates on feature_set" begin
    # Without feature_set in the hash, two params differing only on feature_set would collide
    # in the compiler cache and silently return a cubin compiled for the wrong feature set.
    base = CUDACore.CUDACompilerParams(sm=sm"9.0",  ptx=v"8.0")
    arch = CUDACore.CUDACompilerParams(sm=sm"9.0a", ptx=v"8.0")
    @test hash(base) != hash(arch)
    @test base != arch
end

@testset "ptx_cap_support" begin
    # Architecture-specific needs CC >= 9.0 (i.e. no `*a` keys below sm"9.0") and PTX >= 8.0.
    # Family-specific needs CC >= 10.0 (no `*f` keys below sm"10.0") and PTX >= 8.8.
    @test !(sm"8.6a" in CUDACore.ptx_cap_support(v"8.0"))   # no `*a` below CC 9.0
    @test !(sm"9.0a" in CUDACore.ptx_cap_support(v"7.8"))   # `*a` requires PTX >= 8.0
    @test !(sm"9.0f" in CUDACore.ptx_cap_support(v"8.0"))   # no `*f` below CC 10.0
    @test !(sm"10.0f" in CUDACore.ptx_cap_support(v"8.7"))  # `*f` requires PTX >= 8.8
    @test  sm"9.0a"  in CUDACore.ptx_cap_support(v"8.0")
    @test  sm"10.0f" in CUDACore.ptx_cap_support(v"8.8")
    @test  sm"5.0"   in CUDACore.ptx_cap_support(v"6.2")
end

@testset "llvm_cap_support" begin
    # Floors come from `def : Proc<"sm_NNa", ...>` etc. in NVPTX.td.
    @test  sm"10.3a" in CUDACore.llvm_cap_support(v"21")
    @test  sm"10.0f" in CUDACore.llvm_cap_support(v"21")
    @test  sm"9.0a"  in CUDACore.llvm_cap_support(v"18")
    @test !(sm"9.0a"  in CUDACore.llvm_cap_support(v"17"))  # sm_90a added in LLVM 18
    @test !(sm"10.3a" in CUDACore.llvm_cap_support(v"20"))  # sm_103a added in LLVM 21
    @test !(sm"10.0f" in CUDACore.llvm_cap_support(v"20"))  # sm_100f added in LLVM 21
    @test  sm"7.0"   in CUDACore.llvm_cap_support(v"15")
end

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
