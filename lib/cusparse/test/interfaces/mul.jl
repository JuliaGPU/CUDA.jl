using cuSPARSE
using LinearAlgebra, SparseArrays

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    k = 15
    n = 10
    nB = 2
    alpha = rand(elty)
    beta  = rand(elty)

    if elty <: Real
        eltya = elty
        eltyb = complex(elty)
        @testset "CuSparseMatrix * CuVector -- mul!(c, A, b) mixed ($eltya, $eltyb) opa = $opa" for opa in (identity, transpose, adjoint)
            A = opa == identity ? sprand(eltya, n, m, 0.5) : sprand(eltya, m, n, 0.5)
            b = rand(eltyb, m)
            c = rand(eltyb, n)

            dA = CuSparseMatrixCSR(A)
            db = CuArray(b)
            dc = CuArray(c)

            mul!(c, opa(A), b)
            mul!(dc, opa(dA), db)
            @test c ≈ collect(dc)
        end
    end

    @testset "CuMatrix ops opa = $opa" for opa in (identity, transpose, adjoint)
        A_mv  = opa == identity ? rand(elty, n, m) : rand(elty, m, n)
        dA_mv = CuArray(A_mv)
        A     = opa == identity ? rand(elty, m, k) : rand(elty, k, m)
        dA    = CuArray(A)
        @testset "CuMatrix * CuSparseVector" begin
            elty <: Complex && opa == adjoint && continue
            b_vec  = sprand(elty, m, 0.5)
            db_vec = CuSparseVector(b_vec)
            c_vec  = rand(elty, n)
            dc_vec = CuArray(c_vec)
            @testset "A * b" begin
                c  = opa(A_mv) * b_vec
                dc = opa(dA_mv) * db_vec
                @test c ≈ collect(dc)
            end
            @testset "mul!(c, A, b)" begin
                mul!(c_vec, opa(A_mv), b_vec, alpha, beta)
                mul!(dc_vec, opa(dA_mv), db_vec, alpha, beta)
                @test c_vec ≈ collect(dc_vec)
            end
        end
        @testset "opb = $opb" for opb in (identity, transpose, adjoint)
            B  = opb == identity ? sprand(elty, k, n, 0.2) : sprand(elty, n, k, 0.2)
            for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
                dB = SparseMatrixType(B)
                if SparseMatrixType == CuSparseMatrixCOO
                    dB = sort_coo(dB, 'C')
                end
                @testset "CuMatrix * $SparseMatrixType" begin
                    @testset "A * B" begin
                        C  = opa(A) * opb(B)
                        dC = opa(dA) * opb(dB)
                        @test C ≈ collect(dC)
                    end
                    @testset "mul!(C, A, B)" begin
                        C  = rand(elty, m, n)
                        dC = CuArray(C)
                        mul!(C, opa(A), opb(B), alpha, beta)
                        mul!(dC, opa(dA), opb(dB), alpha, beta)
                        @test C ≈ collect(dC)
                    end
                end
            end
        end
    end

    @testset "Sparse matrix ops opa = $opa" for opa in (identity, transpose, adjoint)
        geam_A = opa == identity ? sprand(elty, n, m, 0.2) : sprand(elty, m, n, 0.2)
        A      = opa == identity ? sprand(elty, m, k, 0.2) : sprand(elty, k, m, 0.2)
        b_vec  = rand(elty, m)
        db_vec = CuArray(b_vec)
        b_spvec = sprand(elty, m, 0.5)
        db_spvec = CuSparseVector(b_spvec)
        @testset "opb = $opb" for opb in (identity, transpose, adjoint)
            geam_B   = opb == identity ? sprand(elty, n, m, 0.5) : sprand(elty, m, n, 0.5)
            B        = opb == identity ? sprand(elty, k, n, 0.5) : sprand(elty, n, k, 0.5)
            B_dense  = opb == identity ? rand(elty, k, n) : rand(elty, n, k)
            dB_dense = CuArray(B_dense)
            @testset "type = $SparseMatrixType" for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
                dA = (SparseMatrixType == CuSparseMatrixBSR) ? SparseMatrixType(A,1) : SparseMatrixType(A)
                dB = (SparseMatrixType == CuSparseMatrixBSR) ? SparseMatrixType(B,1) : SparseMatrixType(B)
                if SparseMatrixType != CuSparseMatrixBSR
                    d_geam_A = SparseMatrixType(geam_A)
                    d_geam_B = SparseMatrixType(geam_B)
                    @testset "A ± B" begin
                        C  = opa(geam_A) + opb(geam_B)
                        dC = opa(d_geam_A) + opb(d_geam_B)
                        @test C ≈ collect(dC)

                        C  = opa(geam_A) - opb(geam_B)
                        dC = opa(d_geam_A) - opb(d_geam_B)
                        @test C ≈ collect(dC)
                        if opa == opb == identity && SparseMatrixType != CuSparseMatrixCOO
                            C  = alpha * geam_A + beta * geam_B
                            dC = geam(alpha, d_geam_A, beta, d_geam_B, 'O')
                            @test C ≈ collect(dC)
                        end
                    end

                    @testset "A * B" begin
                        C  = opa(A) * opb(B)
                        dC = opa(dA) * opb(dB)
                        @test C ≈ collect(dC)
                        if opa == opb == identity
                            mul!(dC, opa(dA), opb(dB), 3, 3.2)
                            C = 3.2 * C + 3 * opa(A) * opb(B)
                            @test collect(dC) ≈ C
                        end
                    end
                    @testset "A * CuVector" begin
                        @testset "A * b" begin
                            c = opa(geam_A) * b_vec
                            dc = opa(d_geam_A) * db_vec
                            @test c ≈ collect(dc)
                        end
                        @testset "mul!(c, A, b)" begin
                            c = rand(elty, n)
                            dc = CuArray(c)

                            mul!(c, opa(geam_A), b_vec, alpha, beta)
                            mul!(dc, opa(d_geam_A), db_vec, alpha, beta)
                            @test c ≈ collect(dc)
                        end
                    end
                    @testset "A * CuSparseVector" begin
                        @testset "A * b" begin
                            c  = opa(geam_A) * b_spvec
                            dc = opa(d_geam_A) * db_spvec
                            @test c ≈ collect(dc)
                        end
                        @testset "mul!(c, A, b)" begin
                            c = rand(elty, n)
                            dc = CuArray(c)
                            mul!(c, opa(geam_A), b_spvec, alpha, beta)
                            mul!(dc, opa(d_geam_A), db_spvec, alpha, beta)
                            @test c ≈ collect(dc)
                        end
                    end
                end
                # end of BSR exclusion
                @testset "CuSparseMatrix * CuMatrix $elty" begin
                    (cuSPARSE.version() < v"12.5.1") && (SparseMatrixType == CuSparseMatrixBSR) && continue
                    (opa != identity) && (SparseMatrixType == CuSparseMatrixBSR) && continue
                    @testset "A * B" begin
                        C  = opa(A) * opb(B_dense)
                        dC = opa(dA) * opb(dB_dense)
                        @test C ≈ collect(dC)
                    end
                    @testset "mul!(C, A, B)" begin
                        C  = rand(elty, m, n)
                        dC = CuArray(C)
                        mul!(C, opa(A), opb(B_dense), alpha, beta)
                        mul!(dC, opa(dA), opb(dB_dense), alpha, beta)
                        @test C ≈ collect(dC)
                    end
                end
            end
        end
    end

    @testset "A ± HermOrSym(B) (CUDA.jl#3043)" begin
        A = sprand(elty, n, n, 0.3)
        B = sprand(elty, n, n, 0.3)
        @testset "$SparseMatrixType" for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
            dA = SparseMatrixType(A)
            dB = SparseMatrixType(B)
            for wrap in (Symmetric, Hermitian), uplo in (:U, :L),
                opa in (identity, transpose, adjoint), op in (+, -)

                @test collect(op(opa(dA), wrap(dB, uplo))) ≈ op(opa(A), wrap(B, uplo))
                @test collect(op(wrap(dB, uplo), opa(dA))) ≈ op(wrap(B, uplo), opa(A))
            end
        end
    end
end
