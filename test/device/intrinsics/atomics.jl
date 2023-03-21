using CUDA: @atomic, @atomicswap, @atomicreplace

@testset "atomics (high-level)" begin
    # tested on all types supported by atomic_cas! (which empowers the fallback definition)
    
    @testset "add" begin
        types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])
    
        function kernel(T, a)
            @atomic a[1] += 1
            return
        end

        @testset for T in types
            a = CuArray([zero(T)])    
            @cuda threads=1024 kernel(T, a)
            @test Array(a)[1] == 1024
        end
    end
    
    @testset "sub" begin
        types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])
    
        function kernel(T, a)
            @atomic a[1] -= 1
            return
        end

        @testset for T in types
            a = CuArray(T[2048])    
            @cuda threads=1024 kernel(T, a)
            @test Array(a)[1] == 1024
        end
    end
    
    @testset "mul" begin
        types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])
    
        function kernel(T, a)
            @atomic a[1] *= 2
            return
        end

        @testset for T in types
            a = CuArray(T[1])
            @cuda threads=5 kernel(T, a)
            @test Array(a)[1] == 32
        end
    end
    
    @testset "div" begin
        types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])
    
        function kernel(T, a)
            @atomic a[1] /= 2
            return
        end

        @testset for T in types
            a = CuArray(T[32])    
            @cuda threads=5 kernel(T, a)
            @test Array(a)[1] == 1
        end
    end
    
    @testset "and" begin
        types = [Int32, Int64, UInt32, UInt64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])
    
        function kernel(T, a)
            i = threadIdx().x
            mask = ~(T(1) << (i-1))
            @atomic a[1] &= mask
            return
        end
            
        @testset for T in types
            a = CuArray([~zero(T)])    
            @cuda threads=8*sizeof(T) kernel(T, a)
            @test Array(a)[1] == zero(T)
        end
    end
    
    @testset "or" begin
        types = [Int32, Int64, UInt32, UInt64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])

        function kernel(T, a)
            i = threadIdx().x
            mask = T(1) << (i-1)
            @atomic a[1] |= mask
            return
        end

        @testset for T in types
            a = CuArray([zero(T)])    
            @cuda threads=8*sizeof(T) kernel(T, a)
            @test Array(a)[1] == ~zero(T)
        end
    end
    
    @testset "xor" begin
        types = [Int32, Int64, UInt32, UInt64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])
    
        function kernel(T, a)
            i = threadIdx().x
            mask = T(1) << ((i-1)%(8*sizeof(T)))
            @atomic a[1] ⊻= mask
            return
        end

        @testset for T in types
            a = CuArray([zero(T)])
            nb = 4
            @cuda threads=(8*sizeof(T)+nb) kernel(T, a)
            @test Array(a)[1] == ~zero(T) & ~((one(T) << nb) - one(T))
        end
    end
    
    @testset "max" begin
        types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])
    
        function kernel(T, a)
            i = threadIdx().x
            @atomic a[1] max i
            return
        end

        @testset for T in types
            a = CuArray([zero(T)])    
            @cuda threads=32 kernel(T, a)
            @test Array(a)[1] == 32
        end
    end
    
    @testset "min" begin
        types = [Int32, Int64, UInt32, UInt64, Float32, Float64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16, Float16])
    
        function kernel(T, a)
            i = threadIdx().x
            @atomic a[1] min i
            return
        end

        @testset for T in types
            a = CuArray([typemax(T)])
            @cuda threads=32 kernel(T, a)
            @test Array(a)[1] == 1
        end
    end
    
    @testset "shift" begin
        types = [Int32, Int64, UInt32, UInt64]
        # capability(device()) >= v"7.0" && append!(types, [Int16, UInt16])
    
        function kernel(T, a)
            @atomic a[1] <<= 1
            return
        end

        @testset for T in types
            a = CuArray([one(T)])    
            @cuda threads=8 kernel(T, a)
            @test Array(a)[1] == 1<<8
        end
    end
    
    @testset "NaN" begin
        f(x,y) = 3x + 2y

        function kernel(x)
            @inbounds CUDA.@atomic x[1] f 42f0
            nothing
        end

        a = CuArray([0f0])
        @cuda kernel(a)
        @test Array(a)[1] ≈ 84

        a = CuArray([NaN32])
        @cuda kernel(a)
        @test isnan(Array(a)[1])
    end

    @testset "macro" begin
        @test_throws_macro ErrorException("could not parse @atomic expression wat(a[1])") @macroexpand begin
            @atomic wat(a[1])
        end
    
        @test_throws_macro ErrorException("@atomic modify expression missing field access") @macroexpand begin
            @atomic a = a + 1
        end
    end
    
end
    