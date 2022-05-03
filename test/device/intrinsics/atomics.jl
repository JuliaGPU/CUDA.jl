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

    @test_throws_macro AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        @atomic a[1] = 1
    end
    @test_throws_macro AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        @atomic a[1] = b ? 1 : 2
    end

    @test_throws_macro AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side") @macroexpand begin
        @atomic a[1] = a[2] + 1
    end

    @test_throws_macro AtomicError("unknown @atomic expression") @macroexpand begin
        @atomic wat(a[1])
    end

    @test_throws_macro AtomicError("@atomic should be applied to an array reference expression") @macroexpand begin
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
