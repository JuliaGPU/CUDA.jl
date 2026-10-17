using cuSOLVER
using LinearAlgebra

m = 15
n = 10

eltypes = [Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64, Int32, Int64, Complex{Int32}, Complex{Int64}]

# Both inputs are promoted to a single BLAS type before solving, so the full cartesian product
# mostly repeats the same 4 solves. Instead, test every eltype on each side against Float32
# (which reaches all 4 BLAS types), plus pairs that promote to a type neither input has.
# The conversions themselves are covered by CUDA.jl's test/core/array.jl.
eltype_pairs = unique!(vcat(
    [(elty, Float32) for elty in eltypes],
    [(Float32, elty) for elty in eltypes],
    [(Float64, ComplexF32), (ComplexF16, Float64), (Int64, ComplexF16)]
))

@testset "Matrix division $elty1 \\ $elty2" for (elty1, elty2) in eltype_pairs
    @testset "Symmetric linear systems" begin
        A = rand(elty1, n, n)
        A = A + transpose(A)
        B = rand(elty2, n, 5)
        b = rand(elty2, n)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = Symmetric(cublasfloat.(A))
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        # the symmetric solve goes through cuSOLVER's 64-bit generic sytrs
        if cuSOLVER.version() >= v"11"
            @test Array(Symmetric(d_A) \ d_B) ≈ (Af \ Bf)
            @test Array(Symmetric(d_A) \ d_b) ≈ (Af \ bf)
        end
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end

    @testset "Square and unsymmetric linear systems" begin
        A = rand(elty1, n, n)
        B = rand(elty2, n, 5)
        b = rand(elty2, n)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = cublasfloat.(A)
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end

    @testset "Overdetermined linear systems" begin
        A = rand(elty1, m, n)
        B = rand(elty2, m, 5)
        b = rand(elty2, m)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = cublasfloat.(A)
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end

    @testset "Underdetermined linear systems" begin
        A = rand(elty1, n, m)
        B = rand(elty2, n, 5)
        b = rand(elty2, n)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = cublasfloat.(A)
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end
end
