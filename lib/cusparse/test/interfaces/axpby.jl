using cuSPARSE
using LinearAlgebra, SparseArrays

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    n = 10
    alpha = rand(elty)
    beta  = rand(elty)

    @testset "CuSparseVector -- axpby -- A ± B" begin
        A  = sprand(elty, n, 0.3)
        B  = sprand(elty, n, 0.7)

        dA = CuSparseVector(A)
        dB = CuSparseVector(B)

        C  = alpha * A + beta * B
        dC = axpby(alpha, dA, beta, dB, 'O')
        @test C ≈ collect(dC)

        C  = A + B
        dC = dA + dB
        @test C ≈ collect(dC)

        C  = A - B
        dC = dA - dB
        @test C ≈ collect(dC)
    end
end
