# TODO: unify with Base.@atomic
using CUDA: @atomic
using BFloat16s: BFloat16

@testset "atomics (low-level)" begin

# tested on all natively-supported atomics

@testset "atomic_add" begin
    types = [Int32, Int64, UInt32, UInt64, Float32]
    capability(device()) >= v"6.0" && push!(types, Float64)
    capability(device()) >= v"7.0" && push!(types, Float16)

    @testset for T in types
        a = CuArray(T[0])

        function kernel(a, b)
            CUDA.atomic_add!(pointer(a), b)
            return
        end

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == 1024
    end
end

@testset "atomic_sub" begin
    types = [Int32, Int64, UInt32, UInt64]

    @testset for T in types
        a = CuArray(T[2048])

        function kernel(a, b)
            CUDA.atomic_sub!(pointer(a), b)
            return
        end

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == 1024
    end
end

@testset "atomic_inc" begin
    @testset for T in [Int32]
        a = CuArray(T[0])

        function kernel(a, b)
            CUDA.atomic_inc!(pointer(a), b)
            return
        end

        @cuda threads=768 kernel(a, T(512))
        @test Array(a)[1] == 255
    end
end

@testset "atomic_dec" begin
    @testset for T in [Int32]
        a = CuArray(T[1024])

        function kernel(a, b)
            CUDA.atomic_dec!(pointer(a), b)
            return
        end

        @cuda threads=256 kernel(a, T(512))
        @test Array(a)[1] == 257
    end
end

@testset "atomic_xchg" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T)])

        function kernel(a, b)
            CUDA.atomic_xchg!(pointer(a), b)
            return
        end

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == one(T)
    end
end

@testset "atomic_and" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray(T[1023])

        function kernel(a, T)
            i = threadIdx().x - 1
            k = 1
            for i = 1:i
                k *= 2
            end
            b = 1023 - k  # 1023 - 2^i
            CUDA.atomic_and!(pointer(a), T(b))
            return
        end

        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == zero(T)
    end
end

@testset "atomic_or" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray(T[0])

        function kernel(a, T)
            i = threadIdx().x
            b = 1  # 2^(i-1)
            for i = 1:i
                b *= 2
            end
            b /= 2
            CUDA.atomic_or!(pointer(a), T(b))
            return
        end

        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == 1023
    end
end

@testset "atomic_xor" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray(T[1023])

        function kernel(a, T)
            i = threadIdx().x
            b = 1  # 2^(i-1)
            for i = 1:i
                b *= 2
            end
            b /= 2
            CUDA.atomic_xor!(pointer(a), T(b))
            return
        end

        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == 0
    end
end

@testset "atomic_cas" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"7.0" && append!(types, [UInt16, BFloat16])

    @testset for T in types
        a = CuArray(T[0])

        function kernel(a, b, c)
            CUDA.atomic_cas!(pointer(a), b, c)
            return
        end

        @cuda threads=1024 kernel(a, zero(T), one(T))
        @test Array(a)[1] == 1
    end
end

@testset "atomic_max" begin
    types = [Int32, Int64, UInt32, UInt64]

    @testset for T in types
        a = CuArray([zero(T)])

        function kernel(a, T)
            i = threadIdx().x
            CUDA.atomic_max!(pointer(a), T(i))
            return
        end

        @cuda threads=1024 kernel(a, T)
        @test Array(a)[1] == 1024
    end
end

@testset "atomic_min" begin
    types = [Int32, Int64, UInt32, UInt64]

    @testset for T in types
        a = CuArray(T[1024])

        function kernel(a, T)
            i = threadIdx().x
            CUDA.atomic_min!(pointer(a), T(i))
            return
        end

        @cuda threads=1024 kernel(a, T)
        @test Array(a)[1] == 1
    end
end

@testset "shared memory" begin
    function kernel()
        shared = CuStaticSharedArray(Float32, 1)
        @atomic shared[threadIdx().x] += 0f0
        return
    end

    CUDA.@sync @cuda kernel()
end

end

@testset "atomics (high-level)" begin

# tested on all types supported by atomic_cas! (which empowers the fallback definition)

@testset "add" begin
    types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])

    @testset for T in types
        a = CuArray([zero(T)])

        function kernel(T, a)
            @atomic a[1] = a[1] + 1
            @atomic a[1] += 1
            return
        end

        @cuda threads=1024 kernel(T, a)
        @test Array(a)[1] == 2048
    end
end

@testset "sub" begin
    types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])

    @testset for T in types
        a = CuArray(T[2048])

        function kernel(T, a)
            @atomic a[1] = a[1] - 1
            @atomic a[1] -= 1
            return
        end

        @cuda threads=1024 kernel(T, a)
        @test Array(a)[1] == 0
    end
end

@testset "mul" begin
    types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])

    @testset for T in types
        a = CuArray(T[1])

        function kernel(T, a)
            @atomic a[1] = a[1] * 2
            @atomic a[1] *= 2
            return
        end

        @cuda threads=5 kernel(T, a)
        @test Array(a)[1] == 1024
    end
end

@testset "div" begin
    types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])

    @testset for T in types
        a = CuArray(T[1024])

        function kernel(T, a)
            @atomic a[1] = a[1] / 2
            @atomic a[1] /= 2
            return
        end

        @cuda threads=5 kernel(T, a)
        @test Array(a)[1] == 1
    end
end

@testset "and" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])

    @testset for T in types
        a = CuArray([~zero(T), ~zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            mask = ~(T(1) << (i-1))
            @atomic a[1] = a[1] & mask
            @atomic a[2] &= mask
            return
        end

        @cuda threads=8*sizeof(T) kernel(T, a)
        @test Array(a)[1] == zero(T)
        @test Array(a)[2] == zero(T)
    end
end

@testset "or" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])

    @testset for T in types
        a = CuArray([zero(T), zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            mask = T(1) << (i-1)
            @atomic a[1] = a[1] | mask
            @atomic a[2] |= mask
            return
        end

        @cuda threads=8*sizeof(T) kernel(T, a)
        @test Array(a)[1] == ~zero(T)
        @test Array(a)[2] == ~zero(T)
    end
end

@testset "xor" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])

    @testset for T in types
        a = CuArray([zero(T), zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            mask = T(1) << ((i-1)%(8*sizeof(T)))
            @atomic a[1] = a[1] ⊻ mask
            @atomic a[2] ⊻= mask
            return
        end

        nb = 4
        @cuda threads=(8*sizeof(T)+nb) kernel(T, a)
        @test Array(a)[1] == ~zero(T) & ~((one(T) << nb) - one(T))
        @test Array(a)[2] == ~zero(T) & ~((one(T) << nb) - one(T))
    end
end

@testset "max" begin
    types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])

    @testset for T in types
        a = CuArray([zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            @atomic a[1] = max(a[1], i)
            return
        end

        @cuda threads=32 kernel(T, a)
        @test Array(a)[1] == 32
    end
end

@testset "min" begin
    types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])

    @testset for T in types
        a = CuArray([typemax(T)])

        function kernel(T, a)
            i = threadIdx().x
            @atomic a[1] = min(a[1], i)
            return
        end

        @cuda threads=32 kernel(T, a)
        @test Array(a)[1] == 1
    end
end

@testset "shift" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])

    @testset for T in types
        a = CuArray([one(T)])

        function kernel(T, a)
            @atomic a[1] <<= 1
            return
        end

        @cuda threads=8 kernel(T, a)
        @test Array(a)[1] == 1<<8
    end
end

@testset "macro" begin

    @testset "NaN" begin
        f(x,y) = 3x + 2y

        function kernel(x)
            CUDA.@atomic x[1] = f(x[1],42f0)
            nothing
        end

        a = CuArray([0f0])
        @cuda kernel(a)
        @test Array(a)[1] ≈ 84

        a = CuArray([NaN32])
        @cuda kernel(a)
        @test isnan(Array(a)[1])
    end

    using CUDA: AtomicError

    @test_throws AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        @atomic a[1] = 1
    end
    @test_throws AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        @atomic a[1] = b ? 1 : 2
    end

    @test_throws AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side") @macroexpand begin
        @atomic a[1] = a[2] + 1
    end

    @test_throws AtomicError("unknown @atomic expression") @macroexpand begin
        @atomic wat(a[1])
    end

    @test_throws AtomicError("@atomic should be applied to an array reference expression") @macroexpand begin
        @atomic a = a + 1
    end
end

@testset "shared memory" begin
    # test that atomic operations on shared memory work
    # https://github.com/JuliaGPU/CUDA.jl/issues/311

    function kernel(a)
        b = CUDA.CuStaticSharedArray(Int, 1)

        if threadIdx().x == 1
            b[] = a[]
        end
        sync_threads()

        CUDA.atomic_add!(pointer(b), 1)
        sync_threads()

        if threadIdx().x == 1
            a[] = b[]
        end
        return
    end

    a = CuArray([0])
    @cuda threads=16 kernel(a)
    @test Array(a) == [16]
end

@testset "shared memory bug" begin
    # shared memory atomics resulted in illegal memory accesses
    # https://github.com/JuliaGPU/CUDA.jl/issues/558

    function kernel()
        tid = threadIdx().x
        shared = CuStaticSharedArray(Float32, 4)
        CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
        sync_threads()
        CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
        return
    end

    @cuda threads=2 kernel()
    synchronize()
end

end


@testset "memory scopes" begin

dev_cap = capability(device())
system_scope_supported = dev_cap >= v"6.0" &&
                         (!Sys.iswindows() || dev_cap >= v"7.0") &&
                         (!CUDA.is_tegra() || dev_cap >= v"7.2")

@testset "reflection" begin
    # what LLVM spells out depends on the target: sm_5x has no scope qualifiers at all,
    # sm_6x adds them, and sm_70+ also adds acquire/release semantics.
    function cas_pattern(cap, scope)
        sem = cap >= v"7.0" ? ".acq_rel" : ""
        qual = cap >= v"6.0" ? ".$scope" : ""
        "atom$sem$qual.global.cas.b32"
    end

    # atomicrmw carries the scope in the IR; whether the back-end spells it out in PTX
    # depends on the LLVM version, so check the IR rather than the PTX.
    @test @filecheck CUDA.code_llvm(Tuple{CuDeviceVector{Int32,1}}) do a
        @check "atomicrmw add {{.*}} syncscope(\"device\")"
        CUDA.atomic_add!(pointer(a), Int32(1))
        return
    end
    @test @filecheck CUDA.code_llvm(Tuple{CuDeviceVector{Int32,1}}) do a
        @check "atomicrmw add {{.*}} syncscope(\"block\")"
        CUDA.atomic_add!(pointer(a), Int32(1), Val(:block))
        return
    end
    @test @filecheck CUDA.code_llvm(Tuple{CuDeviceVector{Int32,1}}) do a
        @check "atomicrmw add"
        @check_not "syncscope"
        CUDA.atomic_add!(pointer(a), Int32(1), Val(:system))
        return
    end

    for (arch, cap) in ((nothing, dev_cap), (sm"61", v"6.1"), (sm"50", v"5.0"))
        kwargs = arch === nothing ? (;) : (; arch)
        @test @filecheck CUDA.code_ptx(Tuple{CuDeviceVector{Int32,1}}; kwargs...) do a
            @check cas_pattern(cap, "gpu")
            CUDA.atomic_cas!(pointer(a), Int32(0), Int32(1))
            return
        end
        @test @filecheck CUDA.code_ptx(Tuple{CuDeviceVector{Int32,1}}; kwargs...) do a
            @check cas_pattern(cap, "cta")
            CUDA.atomic_cas!(pointer(a), Int32(0), Int32(1), Val(:block))
            return
        end
        if cap >= v"6.0"
            @test @filecheck CUDA.code_ptx(Tuple{CuDeviceVector{Int32,1}}; kwargs...) do a
                @check cas_pattern(cap, "sys")
                CUDA.atomic_cas!(pointer(a), Int32(0), Int32(1), Val(:system))
                return
            end
        end
    end

    # Bounds-checking exception paths must not introduce system atomics (#3187).
    function checked_store(a)
        a[1] = Int32(42)
        return
    end
    ptx = sprint(io -> CUDA.code_ptx(io, checked_store, Tuple{CuDeviceVector{Int32,1}};
                                    arch=sm"61", ptx=v"8.8", kernel=true, dump_module=true))
    @test occursin("atom.gpu", ptx)
    @test !occursin(r"(?:atom|red)\.sys", ptx)

    if dev_cap >= v"7.0"
        # 16-bit CAS uses inline assembly, which spells out the scope too
        @test @filecheck CUDA.code_ptx(Tuple{CuDeviceVector{Int16,1}}) do a
            @check "atom.acq_rel.gpu.global.cas.b16"
            CUDA.atomic_cas!(pointer(a), Int16(0), Int16(1))
            return
        end
        @test @filecheck CUDA.code_ptx(Tuple{CuDeviceVector{Int16,1}}) do a
            @check "atom.acq_rel.cta.global.cas.b16"
            CUDA.atomic_cas!(pointer(a), Int16(0), Int16(1), Val(:block))
            return
        end
    end
end

@testset "block scope" begin
    a = CuArray(Int32[0])

    function add_kernel(a, scope)
        CUDA.atomic_add!(pointer(a), Int32(1), scope)
        return
    end
    @cuda threads=1024 add_kernel(a, Val(:block))
    @test Array(a)[1] == 1024

    function cas_kernel(a, scope)
        CUDA.atomic_cas!(pointer(a), Int32(1024), Int32(1), scope)
        return
    end
    @cuda threads=1024 cas_kernel(a, Val(:block))
    @test Array(a)[1] == 1

    function incdec_kernel(a, scope)
        CUDA.atomic_inc!(pointer(a), Int32(2047), scope)
        CUDA.atomic_dec!(pointer(a, 2), Int32(2047), scope)
        return
    end
    a = CuArray(Int32[0, 0])
    @cuda threads=1024 incdec_kernel(a, Val(:block))
    @test Array(a) == [1024, 2048 - 1024]
end

# system-scope atomics require sm_60, and are not available on Pascal under Windows
if system_scope_supported
@testset "system scope" begin
    function add_kernel(a, scope)
        CUDA.atomic_add!(pointer(a), Int32(1), scope)
        return
    end
    function cas_kernel(a, scope)
        CUDA.atomic_cas!(pointer(a), Int32(1024), Int32(1), scope)
        return
    end
    function incdec_kernel(a, scope)
        CUDA.atomic_inc!(pointer(a), Int32(2047), scope)
        CUDA.atomic_dec!(pointer(a, 2), Int32(2047), scope)
        return
    end

    a = CuArray(Int32[0])
    @cuda threads=1024 add_kernel(a, Val(:system))
    @test Array(a)[1] == 1024
    @cuda threads=1024 cas_kernel(a, Val(:system))
    @test Array(a)[1] == 1

    a = CuArray(Int32[0, 0])
    @cuda threads=1024 incdec_kernel(a, Val(:system))
    @test Array(a) == [1024, 2048 - 1024]

    # Smoke-test host-pinned memory. Synchronization precedes the host read; this
    # does not test concurrent CPU/GPU atomicity.
    counter = Int32[0]
    a = unsafe_wrap(CuArray{Int32,1,CUDA.HostMemory}, counter)
    @cuda threads=1024 add_kernel(a, Val(:system))
    synchronize()
    @test counter[1] == 1024
end
end

@testset "floating-point scopes" begin
    types = [Float32, Float64]
    dev_cap >= v"7.0" && push!(types, Float16)
    for scope in (Val(:block), Val(:device), Val(:system))
        if scope === Val(:system) && !system_scope_supported
            continue
        end
        for T in types
            a = CuArray(T[0])
            function kernel(a, scope)
                CUDA.atomic_add!(pointer(a), one(eltype(a)), scope)
                return
            end
            @cuda threads=128 kernel(a, scope)
            @test Array(a) == T[128]
        end
    end
end

@testset "inc/dec return values and unsigned limits" begin
    scopes = (Val(:device), Val(:block), Val(:system))
    for scope in scopes
        if scope === Val(:system) && !system_scope_supported
            continue
        end
        for op in (CUDA.atomic_inc!, CUDA.atomic_dec!),
            initial in Int32[0, 1, 7, 8, -1, typemin(Int32)],
            limit in Int32[0, 7, -1, typemin(Int32)]
            a = CuArray(Int32[initial])
            result = similar(a)
            function kernel(a, result, op, limit, scope)
                @inbounds result[1] = op(pointer(a), limit, scope)
                return
            end
            @cuda kernel(a, result, op, limit, scope)
            old, bound = reinterpret(UInt32, initial), reinterpret(UInt32, limit)
            expected = if op === CUDA.atomic_inc!
                old >= bound ? UInt32(0) : old + UInt32(1)
            else
                old == 0 || old > bound ? bound : old - UInt32(1)
            end
            @test Array(result) == [initial]
            @test Array(a) == [reinterpret(Int32, expected)]
        end
    end
end

end
