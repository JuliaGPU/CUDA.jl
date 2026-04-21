using cuBLAS
using LinearAlgebra

@testset "conversion argument errors" begin
    @test_throws ArgumentError("Unknown operation D") convert(cuBLAS.cublasOperation_t, 'D')
    @test_throws ArgumentError("Unknown fill mode D") convert(cuBLAS.cublasFillMode_t, 'D')
    @test_throws ArgumentError("Unknown diag mode D") convert(cuBLAS.cublasDiagType_t, 'D')
    @test_throws ArgumentError("Unknown side mode D") convert(cuBLAS.cublasSideMode_t, 'D')
end

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 35

    @testset "trmm!" begin
        alpha = rand(elty)
        A = triu(rand(elty, m, m))
        B = rand(elty, m, n)
        dA = CuArray(A)
        dB = CuArray(B)
        dC = CuArray(zeros(elty, m, n))
        cuBLAS.trmm!('L', 'U', 'N', 'N', alpha, dA, dB, dC)
        @test alpha * A * B ≈ Array(dC)
    end

    @testset "trmm" begin
        alpha = rand(elty)
        A = triu(rand(elty, m, m))
        B = rand(elty, m, n)
        dA = CuArray(A)
        dB = CuArray(B)
        d_C = cuBLAS.trmm('L', 'U', 'N', 'N', alpha, dA, dB)
        @test alpha * A * B ≈ Array(d_C)
    end

    @testset "triangular-dense mul!" begin
        sA = rand(elty, m, m)
        sA = sA + transpose(sA)
        B = rand(elty, m, n)
        C = zeros(elty, m, n)

        for t in (identity, transpose, adjoint),
            TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)

            A = copy(sA) |> TR
            B_L = copy(B)
            B_R = copy(B')
            C_L = copy(C)
            C_R = copy(C')
            dA = CuArray(parent(A)) |> TR
            dB_L = CuArray(parent(B_L))
            dB_R = CuArray(parent(B_R))
            dC_L = CuArray(C_L)
            dC_R = CuArray(C_R)

            D_L = mul!(C_L, t(A), B_L)
            dD_L = mul!(dC_L, t(dA), dB_L)

            D_R = mul!(C_R, B_R, t(A))
            dD_R = mul!(dC_R, dB_R, t(dA))

            @test C_L ≈ Array(dC_L)
            @test D_L ≈ Array(dD_L)
            @test C_R ≈ Array(dC_R)
            @test D_R ≈ Array(dD_R)
        end
    end

    @testset "triangular-triangular mul!" begin
        sA = rand(elty, m, m)
        sA = sA + transpose(sA)
        sB = rand(elty, m, m)
        sB = sB + transpose(sB)
        C0 = zeros(elty, m, m)

        for (TRa, ta, TRb, tb, TRc, a_func, b_func) in (
            (UpperTriangular, identity,  LowerTriangular, identity,  Matrix, triu, tril),
            (LowerTriangular, identity,  UpperTriangular, identity,  Matrix, tril, triu),
            (UpperTriangular, identity,  UpperTriangular, transpose, Matrix, triu, triu),
            (UpperTriangular, transpose, UpperTriangular, identity,  Matrix, triu, triu),
            (LowerTriangular, identity,  LowerTriangular, transpose, Matrix, tril, tril),
            (LowerTriangular, transpose, LowerTriangular, identity,  Matrix, tril, tril),
        )
            A = copy(sA) |> TRa
            B = copy(sB) |> TRb
            C = copy(C0) |> TRc
            dA = CuArray(a_func(parent(sA))) |> TRa
            dB = CuArray(b_func(parent(sB))) |> TRb
            dC = if TRc == Matrix
                CuArray(C0) |> DenseCuMatrix
            else
                CuArray(C0) |> TRc
            end

            D = mul!(C, ta(A), tb(B))
            dD = mul!(dC, ta(dA), tb(dB))

            @test C ≈ Array(dC)
            @test D ≈ Array(dD)
        end
    end
end
