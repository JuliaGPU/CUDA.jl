using cuSOLVER
using LinearAlgebra

m = 15
n = 10

@testset "Matrix division $elty1 \\ $elty2" for elty1 in [
    Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64, Int32, Int64, Complex{Int32}, Complex{Int64}
], elty2 in [
    Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64, Int32, Int64, Complex{Int32}, Complex{Int64}
]
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
        @test Array(Symmetric(d_A) \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @test Array(Symmetric(d_A) \ d_b) ≈ (Af \ bf)
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
