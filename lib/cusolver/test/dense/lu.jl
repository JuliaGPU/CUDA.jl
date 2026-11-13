using cuSOLVER
using LinearAlgebra

m = 15
n = 10

@testset "inv elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "unsymmetric" begin
        A = rand(elty, n, n)
        dA = CuArray(A)
        dA⁻¹ = inv(dA)
        dI = dA * dA⁻¹
        @test Array(dI) ≈ I
    end

    # inv(::Symmetric) goes through cuSOLVER's 64-bit generic sytrs, which requires
    # cuSOLVER 11 or later
    if cuSOLVER.version() >= v"11"
        @testset "symmetric" begin
            A = rand(elty, n, n)
            A = A + transpose(A)
            dA = Symmetric(CuArray(A))
            dA⁻¹ = inv(dA)
            dI = dA.data * dA⁻¹
            @test Array(dI) ≈ I
        end
    end
end

@testset "lu elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = CuArray(rand(elty, m, m))
    F = lu(A)
    @test F.L*F.U ≈ A[F.p, :]

    @test_throws LinearAlgebra.SingularException lu(CUDACore.zeros(elty, n, n))
end

@testset "lu ldiv! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, m)
    B = rand(elty, m, m)
    A_d = CuArray(A)
    B_d = CuArray(B)
    lu_cpu = lu(A)
    lu_gpu = lu(A_d)
    @test ldiv!(lu_cpu, B) ≈ collect(ldiv!(lu_gpu, B_d))
end

@testset "lu large" begin
    A = CuMatrix(rand(1024, 1024))
    lua = lu(A)
    @test Matrix(lua.L) * Matrix(lua.U) ≈ Matrix(lua.P) * Matrix(A)

    A = CuMatrix(rand(1024, 512))
    lua = lu(A)
    @test Matrix(lua.L) * Matrix(lua.U) ≈ Matrix(lua.P) * Matrix(A)

    A = CuMatrix(rand(512, 1024))
    lua = lu(A)
    @test Matrix(lua.L) * Matrix(lua.U) ≈ Matrix(lua.P) * Matrix(A)
end
