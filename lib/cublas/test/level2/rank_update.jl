using cuBLAS
using LinearAlgebra

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

    @testset "ger!" begin
        alpha = rand(elty)
        A = rand(elty, m, m)
        x = rand(elty, m)
        y = rand(elty, m)
        dA = CuArray(A)
        dx = CuArray(x)
        dy = CuArray(y)
        dB = copy(dA)
        cuBLAS.ger!(alpha, dx, dy, dB)
        @test (alpha * x) * y' + A ≈ Array(dB)
    end

    @testset "syr!" begin
        alpha = rand(elty)
        sA = rand(elty, m, m)
        sA = sA + transpose(sA)
        x = rand(elty, m)
        dx = CuArray(x)
        dB = CuArray(sA)
        cuBLAS.syr!('U', alpha, dx, dB)
        B = (alpha * x) * transpose(x) + sA
        @test triu(B) ≈ triu(Array(dB))
    end

    if elty <: Complex
        @testset "her!" begin
            alpha = rand(elty)
            hA = rand(elty, m, m)
            hA = hA + adjoint(hA)
            x = rand(elty, m)
            dx = CuArray(x)
            dB = CuArray(hA)
            cuBLAS.her!('U', real(alpha), dx, dB)
            B = (real(alpha) * x) * x' + hA
            @test triu(B) ≈ triu(Array(dB))
        end

        @testset "her2!" begin
            alpha = rand(elty)
            hA = rand(elty, m, m)
            hA = hA + adjoint(hA)
            x = rand(elty, m)
            y = rand(elty, m)
            dx = CuArray(x)
            dy = CuArray(y)
            dB = CuArray(hA)
            cuBLAS.her2!('U', real(alpha), dx, dy, dB)
            B = (real(alpha) * x) * y' + y * (real(alpha) * x)' + hA
            @test triu(B) ≈ triu(Array(dB))
        end
    end
end
