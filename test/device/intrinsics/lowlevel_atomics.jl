using BFloat16s: BFloat16

function atomic_types(cap)
    types = [
        Int32, Int64, 
        UInt32, UInt64,
        Float64, Float32]
    if cap >= v"6.0"
        append!(types, [
            Int8, Int16,
            UInt8, UInt16,
            Float16])
    end
    return types
end

@testset "atomics (low-level) with order" begin

@testset "atomic_load" begin
    capabilities = (v"3.5", v"6.0", v"7.0")
    current_cap = capability(device())

    capabilities = filter(c->c<=current_cap, capabilities)

    @testset for cap in capabilities
        types = atomic_types(cap)
        scopes = [CUDA.block_scope, CUDA.device_scope, CUDA.system_scope]
        orders = [CUDA.monotonic, CUDA.acquire, CUDA.seq_cst]
        # unsupported_orders = [CUDA.release, CUDA.acq_rel]

        function kernel(a, order, scope)
            CUDA.atomic_load(pointer(a), order, scope)
            return
        end

        @testset for (T, order, scope) in Iterators.product(types, orders, scopes)
            a = CuArray(T[0])
            @cuda cap=cap threads=1 kernel(a, order, scope)
            @test Array(a)[1] == 0
        end
    end
end

@testset "atomic_store!" begin
    if capability(device()) >= v"6.0"
        types = [Int8, Int16, Int32, Int64, 
                 UInt8, UInt16, UInt32, UInt64,
                 Float64, Float32]
        scopes = [CUDA.block_scope, CUDA.device_scope, CUDA.system_scope]
        # TODO unordered
        supported_orders = [CUDA.monotonic, CUDA.release, CUDA.seq_cst]
        unsupported_orders = [CUDA.acquire, CUDA.acq_rel]

        function kernel(a, val, order, scope)
            CUDA.atomic_store!(pointer(a), val, order, scope)
            return
        end

        @testset for (T, order, scope) in Iterators.product(types, supported_orders, scopes)
            a = CuArray(T[0])
            @cuda threads=1 kernel(a, one(T), order, scope)
            @test Array(a)[1] == one(T)
        end
    end
end

@testset "atomic_cas!" begin
    if capability(device()) >= v"6.0"
        # TODO size(T) in (1, 2)
        types = [Int32, Int64, 
                 UInt32, UInt64,
                 Float64, Float32]
        scopes = [CUDA.block_scope, CUDA.device_scope, CUDA.system_scope]
        # TODO unordered
        orders = [CUDA.monotonic, CUDA.release, CUDA.seq_cst, CUDA.acquire, CUDA.acq_rel]

        function kernel(a, expected, desired, success_order, failure_order, scope)
            CUDA.atomic_cas!(pointer(a), expected, desired, success_order, failure_order, scope)
            return
        end

        @testset for (T, success_order, failure_order, scope) in Iterators.product(types, orders, orders, scopes)
            a = CuArray(T[0])
            @cuda threads=1 kernel(a, zero(T), one(T), success_order, failure_order, scope)
            @test Array(a)[1] == one(T)
        end
    end
end

end # atomics (low-level) with order

@testset "atomics (low-level)" begin

# tested on all natively-supported atomics

@testset "atomic_add" begin
    types = [Int32, Int64, UInt32, UInt64, Float32]
    capability(device()) >= v"6.0" && push!(types, Float64)
    capability(device()) >= v"7.0" && push!(types, Float16)

    function kernel(a, b)
        CUDA.atomic_add!(pointer(a), b)
        return
    end

    @testset for T in types
        a = CuArray(T[0])

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == 1024
    end
end

@testset "atomic_sub" begin
    types = [Int32, Int64, UInt32, UInt64]

    function kernel(a, b)
        CUDA.atomic_sub!(pointer(a), b)
        return
    end

    @testset for T in types
        a = CuArray(T[2048])
        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == 1024
    end
end

@testset "atomic_inc" begin
    function kernel(a, b)
        CUDA.atomic_inc!(pointer(a), b)
        return
    end

    @testset for T in [Int32]
        a = CuArray(T[0])
        @cuda threads=768 kernel(a, T(512))
        @test Array(a)[1] == 255
    end
end

@testset "atomic_dec" begin
    function kernel(a, b)
        CUDA.atomic_dec!(pointer(a), b)
        return
    end

    @testset for T in [Int32]
        a = CuArray(T[1024])
        @cuda threads=256 kernel(a, T(512))
        @test Array(a)[1] == 257
    end
end

@testset "atomic_xchg" begin
    function kernel(a, b)
        CUDA.atomic_xchg!(pointer(a), b)
        return
    end
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T)])
        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == one(T)
    end
end

@testset "atomic_and" begin
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
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray(T[1023])
        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == zero(T)
    end
end

@testset "atomic_or" begin
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
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray(T[0])
        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == 1023
    end
end

@testset "atomic_xor" begin
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
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray(T[1023])
        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == 0
    end
end

@testset "atomic_cas" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"7.0" && append!(types, [UInt16, BFloat16])

    function kernel(a, b, c)
        CUDA.atomic_cas!(pointer(a), b, c)
        return
    end

    @testset for T in types
        a = CuArray(T[0])
        @cuda threads=1024 kernel(a, zero(T), one(T))
        @test Array(a)[1] == 1
    end
end

@testset "atomic_max" begin
    types = [Int32, Int64, UInt32, UInt64]

    function kernel(a, T)
        i = threadIdx().x
        CUDA.atomic_max!(pointer(a), T(i))
        return
    end

    @testset for T in types
        a = CuArray([zero(T)])
        @cuda threads=1024 kernel(a, T)
        @test Array(a)[1] == 1024
    end
end

@testset "atomic_min" begin
    types = [Int32, Int64, UInt32, UInt64]

    function kernel(a, T)
        i = threadIdx().x
        CUDA.atomic_min!(pointer(a), T(i))
        return
    end

    @testset for T in types
        a = CuArray(T[1024])
        @cuda threads=1024 kernel(a, T)
        @test Array(a)[1] == 1
    end
end

@testset "shared memory" begin
    @testset "simple" begin
        function kernel()
            shared = CuStaticSharedArray(Float32, 1)
            CUDA.atomic_add!(pointer(shared, threadIdx().x), 0f0)
            return
        end

        CUDA.@sync @cuda kernel()
    end

    @testset "shared memory reduction" begin
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

end # low-level atomics
