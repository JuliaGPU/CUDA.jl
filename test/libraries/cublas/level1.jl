using CUDA.CUBLAS
using LinearAlgebra

using BFloat16s
using StaticArrays

@test CUBLAS.version() isa VersionNumber
@test CUBLAS.version().major == CUBLAS.cublasGetProperty(CUDA.MAJOR_VERSION)
@test CUBLAS.version().minor == CUBLAS.cublasGetProperty(CUDA.MINOR_VERSION)
@test CUBLAS.version().patch == CUBLAS.cublasGetProperty(CUDA.PATCH_LEVEL)

m = 20
n = 35
k = 13

@testset "level 1" begin
    @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
        A = CUDA.rand(T, m)
        B = CuArray{T}(undef, m)
        CUBLAS.copy!(m,A,B)
        @test Array(A) == Array(B)

        @test testf(rmul!, rand(T, 6, 9, 3), rand())
        @test testf(dot, rand(T, m), rand(T, m))
        @test testf(*, transpose(rand(T, m)), rand(T, m))
        @test testf(*, rand(T, m)', rand(T, m))
        @test testf(norm, rand(T, m))
        @test testf(BLAS.asum, rand(T, m))

        @test testf(axpy!, rand(), rand(T, m), rand(T, m))
        @test testf(LinearAlgebra.axpby!, rand(), rand(T, m), rand(), rand(T, m))
        if T <: Complex
            @test testf(dot, rand(T, m), rand(T, m))
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            dz = dot(dx, dy)
            z = dot(x, y)
            @test dz ≈ z
        end

        @testset "rotate!" begin
            @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
            @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
        end
        @testset "reflect!" begin
            @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
            @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
        end

        @testset "rotg!" begin
            a = rand(T)
            b = rand(T)
            a_copy = copy(a)
            b_copy = copy(b)
            a, b, c, s = CUBLAS.rotg!(a, b)
            rot = [c s; -conj(s) c] * [a_copy; b_copy]
            @test rot ≈ [a; 0]
            if T <: Real
                @test a^2 ≈ a_copy^2 + b_copy^2
            end
            @test c^2 + abs2(s) ≈ one(T)
        end

        if T <: Real
            H = rand(T, 2, 2)
            @testset "flag $flag" for (flag, flag_H) in ((T(-2), [one(T) zero(T); zero(T) one(T)]),
                                                         (-one(T), H),
                                                         (zero(T), [one(T) H[1,2]; H[2, 1] one(T)]),
                                                         (one(T), [H[1,1] one(T); -one(T) H[2, 2]]),
                                                        )
                @testset "rotm!" begin
                    rot_n = 2
                    x = rand(T, rot_n)
                    y = rand(T, rot_n)
                    dx = CuArray(x)
                    dy = CuArray(y)
                    dx, dy = CUBLAS.rotm!(rot_n, dx, dy, CuArray(vcat(flag, H...)))
                    h_x = collect(dx)
                    h_y = collect(dy)
                    @test h_x ≈ [x[1] * flag_H[1,1] + y[1] * flag_H[1,2]; x[2] * flag_H[1, 1] + y[2] * flag_H[1, 2]]
                    @test h_y ≈ [x[1] * flag_H[2,1] + y[1] * flag_H[2,2]; x[2] * flag_H[2, 1] + y[2] * flag_H[2, 2]]
                end
            end
            @testset "rotmg!" begin
                gpu_param = CuArray{T}(undef, 5)
                x1 = rand(T)
                y1 = rand(T)
                d1 = zero(T)
                d2 = zero(T)
                x1_copy = copy(x1)
                y1_copy = copy(y1)
                d1, d2, x1, y1 = CUBLAS.rotmg!(d1, d2, x1, y1, gpu_param)
                cpu_param = Array(gpu_param)
                flag = cpu_param[1]
                H = zeros(T, 2, 2)
                if flag == -2
                    H[1, 1] = one(T)
                    H[1, 2] = zero(T)
                    H[2, 1] = zero(T)
                    H[2, 2] = one(T)
                elseif flag == -1
                    H[1, 1] = cpu_param[2]
                    H[1, 2] = cpu_param[3]
                    H[2, 1] = cpu_param[4]
                    H[2, 2] = cpu_param[5]
                elseif iszero(flag)
                    H[1, 1] = one(T)
                    H[1, 2] = cpu_param[3]
                    H[2, 1] = cpu_param[4]
                    H[2, 2] = one(T)
                elseif flag == 1
                    H[1, 1] = cpu_param[2]
                    H[1, 2] = one(T)
                    H[2, 1] = -one(T)
                    H[2, 2] = cpu_param[5]
                end
                out = H * [(√d1) * x1_copy; (√d2) * y1_copy]
                @test out[2] ≈ zero(T)
            end
        end

        @testset "swap!" begin
            # swap is an extension
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            CUBLAS.swap!(m, dx, dy)
            h_x = collect(dx)
            h_y = collect(dy)
            @test h_x ≈ y
            @test h_y ≈ x
        end

        @testset "iamax/iamin" begin
            a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
            ca = CuArray(a)
            @test BLAS.iamax(a) == CUBLAS.iamax(ca)
            @test CUBLAS.iamin(ca) == 3
            result_type = CUBLAS.version() >= v"12.0" ? Int64 : Cint
            result = CuRef{result_type}(0)
            CUBLAS.iamax(ca, result)
            @test BLAS.iamax(a) == result[]
        end
        @testset "nrm2 with result" begin
            x = rand(T, m)
            dx = CuArray(x)
            result = CuRef{real(T)}(zero(real(T)))
            CUBLAS.nrm2(dx, result)
            @test norm(x) ≈ result[]
        end
    end # level 1 testset
    @testset for T in [Float16, ComplexF16]
        A = CuVector(rand(T, m)) # CUDA.rand doesn't work with 16 bit types yet
        B = CuArray{T}(undef, m)
        CUBLAS.copy!(m,A,B)
        @test Array(A) == Array(B)

        @test testf(rmul!, rand(T, 6, 9, 3), rand())
        @test testf(dot, rand(T, m), rand(T, m))
        @test testf(*, transpose(rand(T, m)), rand(T, m))
        @test testf(*, rand(T, m)', rand(T, m))
        @test testf(norm, rand(T, m))
        @test testf(axpy!, rand(), rand(T, m), rand(T, m))
        @test testf(LinearAlgebra.axpby!, rand(), rand(T, m), rand(), rand(T, m))

        if T <: Complex
            @test testf(dot, rand(T, m), rand(T, m))
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            dz = dot(dx, dy)
            z = dot(x, y)
            @test dz ≈ z
        end
    end
end # level 1 testset
