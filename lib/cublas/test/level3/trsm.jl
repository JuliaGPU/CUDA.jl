using cuBLAS
using LinearAlgebra

using Adapt: adapt

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 35

    @testset "trsm adjtype=$adjtype, uplotype=$uplotype" for
        adjtype in (identity, adjoint, transpose),
        uplotype in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)

        A = triu(rand(elty, m, m))
        dA = CuArray(A)
        Br = rand(elty, m, n)
        Bl = rand(elty, n, m)
        d_Br = CuArray(Br)
        d_Bl = CuArray(Bl)
        @test adjtype(uplotype(A)) \ Br ≈ Array(adjtype(uplotype(dA)) \ d_Br)
        @test Bl / adjtype(uplotype(A)) ≈ Array(d_Bl / adjtype(uplotype(dA)))
    end

    @testset "trsm alpha scaling" begin
        alpha = rand(elty)
        A = triu(rand(elty, m, m))
        dA = CuArray(A)
        Br = rand(elty, m, n)
        d_Br = CuArray(Br)
        @test BLAS.trsm('L', 'U', 'N', 'N', alpha, A, Br) ≈ Array(cuBLAS.trsm('L', 'U', 'N', 'N', alpha, dA, d_Br))
    end

    @testset "trsm_batched!" begin
        alpha = rand(elty)
        bA = [triu(rand(elty, m, m)) for i in 1:10]
        bB = [rand(elty, m, n) for i in 1:10]
        bBbad = [rand(elty, m, n) for i in 1:9]
        bd_A = CuArray{elty, 2}[CuArray(a) for a in bA]
        bd_B = CuArray{elty, 2}[CuArray(b) for b in bB]
        bd_Bbad = CuArray{elty, 2}[CuArray(b) for b in bBbad]

        cuBLAS.trsm_batched!('L', 'U', 'N', 'N', alpha, bd_A, bd_B)
        @test_throws DimensionMismatch cuBLAS.trsm_batched!('L', 'U', 'N', 'N', alpha, bd_A, bd_Bbad)
        for i in 1:length(bd_B)
            @test alpha * (bA[i] \ bB[i]) ≈ Array(bd_B[i])
        end
    end

    @testset "trsm_batched" begin
        alpha = rand(elty)
        bA = [triu(rand(elty, m, m)) for i in 1:10]
        bB = [rand(elty, m, n) for i in 1:10]
        bd_A = CuArray{elty, 2}[CuArray(a) for a in bA]
        bd_B = CuArray{elty, 2}[CuArray(b) for b in bB]

        bd_C = cuBLAS.trsm_batched('L', 'U', 'N', 'N', alpha, bd_A, bd_B)
        for i in 1:length(bd_C)
            @test alpha * (bA[i] \ bB[i]) ≈ Array(bd_C[i])
        end
    end

    @testset "left trsm!" begin
        alpha = rand(elty)
        A = triu(rand(elty, m, m))
        B = rand(elty, m, n)
        dA = CuArray(A)
        dB = CuArray(B)
        dC = copy(dB)
        cuBLAS.trsm!('L', 'U', 'N', 'N', alpha, dA, dC)
        @test alpha * (A \ B) ≈ Array(dC)
    end

    @testset "left trsm ($op)" for (op, trans) in ((identity, 'N'), (adjoint, 'C'), (transpose, 'T'))
        alpha = rand(elty)
        A = triu(rand(elty, m, m))
        B = rand(elty, m, n)
        dA = CuArray(A)
        dB = CuArray(B)
        dC = cuBLAS.trsm('L', 'U', trans, 'N', alpha, dA, dB)
        @test alpha * (op(A) \ B) ≈ Array(dC)
    end

    @testset "triangular ldiv!" begin
        A = triu(rand(elty, m, m))
        B = rand(elty, m, m)
        dA = CuArray(A)
        dB = CuArray(B)

        for t in (identity, transpose, adjoint),
            TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)

            dC = copy(dB)
            ldiv!(t(TR(dA)), dC)
            @test t(TR(A)) \ B ≈ Array(dC)
        end
    end

    @testset "right trsm!" begin
        alpha = rand(elty)
        A = rand(elty, m, m)
        B = triu(rand(elty, m, m))
        dA = CuArray(A)
        dB = CuArray(B)
        dC = copy(dA)
        cuBLAS.trsm!('R', 'U', 'N', 'N', alpha, dB, dC)
        @test alpha * (A / B) ≈ Array(dC)
    end

    @testset "right trsm ($op)" for (op, trans) in ((identity, 'N'), (adjoint, 'C'), (transpose, 'T'))
        alpha = rand(elty)
        A = rand(elty, m, m)
        B = triu(rand(elty, m, m))
        dA = CuArray(A)
        dB = CuArray(B)
        dC = cuBLAS.trsm('R', 'U', trans, 'N', alpha, dB, dA)
        @test alpha * (A / op(B)) ≈ Array(dC)
    end

    @testset "triangular rdiv!" begin
        A = rand(elty, m, m)
        B = triu(rand(elty, m, m))
        dA = CuArray(A)
        dB = CuArray(B)

        for t in (identity, transpose, adjoint),
            TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)

            dC = copy(dA)
            rdiv!(dC, t(TR(dB)))
            @test A / t(TR(B)) ≈ Array(dC)
        end
    end

    @testset "Diagonal rdiv!" begin
        A = rand(elty, m, m)
        B = Diagonal(rand(elty, m))
        dA = CuArray(A)
        dB = adapt(CuArray, B)

        d_C = dA / dB
        rdiv!(dA, dB)
        @test A / B ≈ Array(dA)
        @test A / B ≈ Array(d_C)

        B_bad = Diagonal(CuArray(rand(elty, m + 1)))
        @test_throws DimensionMismatch("left hand side has $m columns but D is $(m+1) by $(m+1)") rdiv!(dA, B_bad)
    end
end
