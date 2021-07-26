using SpecialFunctions

@testset "math" begin
    @testset "log10" begin
        @test testf(a->log10.(a), Float32[100])
    end

    @testset "pow" begin
        for T in (Float16, Float32, Float64, ComplexF32, ComplexF64)
            range = (T<:Integer) ? (T(5):T(10)) : T
            @test testf((x,y)->x.^y, rand(Float32, 1), rand(range, 1))
            @test testf((x,y)->x.^y, rand(Float32, 1), -rand(range, 1))
        end
    end

    @testset "isinf" begin
      for x in (Inf32, Inf, NaN32, NaN)
        @test testf(x->isinf.(x), [x])
      end
    end

    @testset "isnan" begin
      for x in (Inf32, Inf, NaN32, NaN)
        @test testf(x->isnan.(x), [x])
      end
    end

    for op in (exp, angle, exp2, exp10,)
        @testset "$op" begin
            for T in (Float16, Float32, Float64)
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end
        end
    end

    for op in (expm1,)
        @testset "$op" begin
            # FIXME: add expm1(::Float16) to Base
            for T in (Float32, Float64)
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end
        end
    end

    for op in (exp, abs, abs2, angle, log)
        @testset "Complex - $op" begin
            for T in (ComplexF16, ComplexF32, ComplexF64)
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end

        end
    end
    @testset "mod and rem" begin
        for T in (Float16, Float32, Float64)
            @test testf(a->rem.(a, T(2)), T[0, 1, 1.5, 2, -1])
            @test testf(a->rem.(a, T(2), RoundNearest), T[0, 1, 1.5, 2, -1])
            @test testf(a->mod.(a, T(2)), T[0, 1, 1.5, 2, -1])
        end
    end

    @testset "rsqrt" begin
        # GPUCompiler.jl#173: a CUDA-only device function fails to validate
        function kernel(a)
            a[] = CUDA.rsqrt(a[])
            return
        end

        # make sure this test uses an actual device function
        @test_throws ErrorException kernel(ones(1))

        for T in (Float16, Float32)
            a = CuArray{T}([4])
            @cuda kernel(a)
            @test Array(a) == [0.5]
        end
    end

    @testset "fma" begin
        for T in (Float16, Float32, Float64)
            @test testf((x,y,z)->fma.(x,y,z), rand(T, 1), rand(T, 1), rand(T, 1))
            @test testf((x,y,z)->fma.(x,y,z), rand(T, 1), -rand(T, 1), -rand(T, 1))
        end
    end

    # something from SpecialFunctions.jl
    @testset "erf" begin
        @test testf(a->SpecialFunctions.erf.(a), Float32[1.0])
    end
end
