using cuBLAS
using LinearAlgebra

@testset for T in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

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
        a, b, c, s = cuBLAS.rotg!(a, b)
        rot = [c s; -conj(s) c] * [a_copy; b_copy]
        @test rot ≈ [a; 0]
        if T <: Real
            @test a^2 ≈ a_copy^2 + b_copy^2
        end
        @test c^2 + abs2(s) ≈ one(T)
    end

    if T <: Real
        H = rand(T, 2, 2)
        @testset "rotm! flag=$flag" for (flag, flag_H) in ((T(-2), [one(T) zero(T); zero(T) one(T)]),
                                                           (-one(T), H),
                                                           (zero(T), [one(T) H[1,2]; H[2, 1] one(T)]),
                                                           (one(T), [H[1,1] one(T); -one(T) H[2, 2]]))
            rot_n = 2
            x = rand(T, rot_n)
            y = rand(T, rot_n)
            dx = CuArray(x)
            dy = CuArray(y)
            dx, dy = cuBLAS.rotm!(rot_n, dx, dy, CuArray(vcat(flag, H...)))
            h_x = collect(dx)
            h_y = collect(dy)
            @test h_x ≈ [x[1] * flag_H[1,1] + y[1] * flag_H[1,2]; x[2] * flag_H[1, 1] + y[2] * flag_H[1, 2]]
            @test h_y ≈ [x[1] * flag_H[2,1] + y[1] * flag_H[2,2]; x[2] * flag_H[2, 1] + y[2] * flag_H[2, 2]]
        end

        @testset "rotmg!" begin
            gpu_param = CuArray{T}(undef, 5)
            x1 = rand(T)
            y1 = rand(T)
            d1 = zero(T)
            d2 = zero(T)
            x1_copy = copy(x1)
            y1_copy = copy(y1)
            d1, d2, x1, y1 = cuBLAS.rotmg!(d1, d2, x1, y1, gpu_param)
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
end
