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
    @testset "loggamma" begin
        @test testf(a->SpecialFunctions.loggamma.(a), Float32[1.0])
    end

    @testset "exp" begin
        # JuliaGPU/CUDA.jl#1085: exp uses Base.sincos performing a global CPU load
        @test testf(x->exp.(x), [1e7im])
    end

    @testset "fastmath" begin
        # libdevice provides some fast math functions
        a(x) = cos(x)
        b(x) = @fastmath cos(x)
        @test Array(map(a, cu([0.1,0.2]))) ≈ Array(map(b, cu([0.1,0.2])))

        # JuliaGPU/CUDA.jl#1352: some functions used to fall back to libm
        f(x) = log1p(x)
        g(x) = @fastmath log1p(x)
        @test Array(map(f, cu([0.1,0.2]))) ≈ Array(map(g, cu([0.1,0.2])))
    end

    @testset "byte_perm" begin
        bytes = UInt32[i for i in 1:8]
        x = bytes[4]<<24 | bytes[3]<<16 | bytes[2]<<8 | bytes[1]<<0
        y = bytes[8]<<24 | bytes[7]<<16 | bytes[6]<<8 | bytes[5]<<0
        sel = UInt32[4, 2, 4, 2]
        code = sel[4]<<12 | sel[3]<<8 | sel[2]<<4 | sel[1]<<0
        r = bytes[sel[4]+1]<<24 | bytes[sel[3]+1]<<16 | bytes[sel[2]+1]<<8 | bytes[sel[1]+1]<<0

        function kernel1(a)
            a[3] = CUDA.byte_perm(a[1], a[2], code % Int32)
            return
        end
        function kernel2(a)
            a[3] = CUDA.byte_perm(a[1], a[2], code % UInt16)
            return
        end

        for T in [UInt32, Int32]
            a = CuArray{T}([x, y, 0])
            @cuda kernel1(a)
            @test Array(a)[3] == r
            a = CuArray{T}([x, y, 0])
            @cuda kernel2(a)
            @test Array(a)[3] == r
        end
    end

    @testset "@fastmath sincos" begin
        # JuliaGPU/CUDA.jl#1606: FastMath.sincos fell back to regular sin/cos
        function kernel(a, b, c)
            @inbounds b[], c[] = @fastmath sincos(a[])
            return
        end
        asm = sprint(io->CUDA.code_ptx(io, kernel, NTuple{3,CuDeviceArray{Float32,1,AS.Global,Int32}}))
        @assert contains(asm, "sin.approx.f32")
        @assert contains(asm, "cos.approx.f32")
        @assert !contains(asm, "__nv")  # from libdevice
    end

    @testset "JuliaGPU/CUDA.jl#2111: min/max should return NaN" begin
        for T in [Float32, Float64]
            AT = CuArray{T}
            @test isequal(Array(min.(AT([NaN]), AT([Inf]))), [NaN])
            @test isequal(minimum(AT([NaN])), NaN)

            @test isequal(Array(max.(AT([NaN]), AT([-Inf]))), [NaN])
            @test isequal(maximum(AT([NaN])), NaN)
        end
    end
end
