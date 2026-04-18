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

    @testset "symmetric" begin
        A = rand(elty, n, n)
        A = A + transpose(A)
        dA = Symmetric(CuArray(A))
        dA⁻¹ = inv(dA)
        dI = dA.data * dA⁻¹
        @test Array(dI) ≈ I
    end

    @testset "triangular" begin
        for (triangle, uplo, diag) in ((LowerTriangular, 'L', 'N'), (UnitLowerTriangular, 'L', 'U'),
                                       (UpperTriangular, 'U', 'N'), (UnitUpperTriangular, 'U', 'U'))
            A = rand(elty, n, n)
            A = uplo == 'L' ? tril(A) : triu(A)
            A = diag == 'N' ? A : A - Diagonal(A) + I
            dA = triangle(view(CuArray(A), 1:2:n, 1:2:n)) # without this view, we are hitting the CUBLAS method!
            dA⁻¹ = inv(dA)
            hI = triangle(Array(parent(dA))) * Array(parent(dA⁻¹))
            @test hI ≈ I
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
