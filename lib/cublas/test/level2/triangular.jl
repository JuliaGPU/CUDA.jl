using cuBLAS
using LinearAlgebra

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

    @testset "trmv!" begin
        A = triu(rand(elty, m, m))
        dA = CuArray(A)
        x = rand(elty, m)
        dx = CuArray(x)
        d_y = copy(dx)
        cuBLAS.trmv!('U', 'N', 'N', dA, d_y)
        @test A * x ≈ Array(d_y)
        @test_throws DimensionMismatch cuBLAS.trmv!('U', 'N', 'N', dA, CuArray(rand(elty, m + 1)))
    end

    @testset "trmv" begin
        A = triu(rand(elty, m, m))
        dA = CuArray(A)
        x = rand(elty, m)
        dx = CuArray(x)
        d_y = cuBLAS.trmv('U', 'N', 'N', dA, dx)
        @test A * x ≈ Array(d_y)
    end

    @testset "lmul!($op($TR))" for TR in (UpperTriangular, LowerTriangular),
                                   op in (identity, adjoint, transpose)
        A = rand(elty, m, m)
        dA = CuArray(A)
        x = rand(elty, m)
        dy = CuArray(x)
        if TR === UpperTriangular && op === identity && VERSION >= v"1.11.2"
            @test istriu(TR(dA))
            @test !istril(TR(dA))
        elseif TR === LowerTriangular && op === identity && VERSION >= v"1.11.2"
            @test !istriu(TR(dA))
            @test istril(TR(dA))
        end
        lmul!(op(TR(dA)), dy)
        y = op(TR(A)) * x
        @test y ≈ Array(dy)
    end

    @testset "trsv!" begin
        A = triu(rand(elty, m, m))
        dA = CuArray(A)
        x = rand(elty, m)
        dx = CuArray(x)
        d_y = copy(dx)
        cuBLAS.trsv!('U', 'N', 'N', dA, d_y)
        @test A \ x ≈ Array(d_y)
        @test_throws DimensionMismatch cuBLAS.trsv!('U', 'N', 'N', dA, CuArray(rand(elty, m + 1)))
    end

    @testset "trsv ($op)" for (op, trans) in ((identity, 'N'), (adjoint, 'C'), (transpose, 'T'))
        A = triu(rand(elty, m, m))
        dA = CuArray(A)
        x = rand(elty, m)
        dx = CuArray(x)
        d_y = cuBLAS.trsv('U', trans, 'N', dA, dx)
        @test op(A) \ x ≈ Array(d_y)
    end

    @testset "ldiv!($op($TR))" for TR in (UpperTriangular, LowerTriangular),
                                   op in (identity, adjoint, transpose)
        A = rand(elty, m, m)
        A = A + transpose(A)
        dA = CuArray(A)
        x = rand(elty, m)
        dx = CuArray(x)
        dy = copy(dx)
        cuBLAS.ldiv!(op(TR(dA)), dy)
        @test op(TR(A)) \ x ≈ Array(dy)
    end

    @testset "inv($TR)" for TR in (UpperTriangular, LowerTriangular,
                                   UnitUpperTriangular, UnitLowerTriangular)
        @test testf(x -> inv(TR(x)), rand(elty, m, m))
    end
end
