using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays

m = 20
k = 15
n = 10
nB = 2

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    alpha = rand(elty)
    beta  = rand(elty)
    @testset "LinearAlgebra" begin
        @testset "CuSparseVector -- axpby -- A ± B" begin 
            A     = sprand(elty, n, 0.3)
            B     = sprand(elty, n, 0.7)

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
                        (CUSPARSE.version() < v"12.5.1") && (SparseMatrixType == CuSparseMatrixBSR) && continue
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
        @testset "ldiv $elty $triangle" for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
            A  = rand(elty, m, m)
            A  = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
            A  = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
            A  = sparse(A)
            x  = rand(elty, m)
            y  = rand(elty, m)
            dy = CuArray(y)
            dx = CuArray(x)
            @testset "opa = $opa" for opa in (identity, transpose, adjoint)
                @testset "type = $SparseMatrixType" for SparseMatrixType in (CuSparseMatrixCOO, CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR)
                    SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    dA = SparseMatrixType == CuSparseMatrixBSR ? CuSparseMatrixBSR(A, 1) : SparseMatrixType(A)
                    @testset "ldiv! -- CuVector" begin
                        z  = rand(elty, m)
                        dz = CuArray(z)
                        ldiv!(triangle(opa(A)), z)
                        ldiv!(triangle(opa(dA)), dz)
                        @test z ≈ collect(dz)
                    end
                    # seems to be a library bug in CUDAs 12.0-12.2, only fp64 types are supported
                    if SparseMatrixType != CuSparseMatrixBSR && (elty ∈ (Float64, ComplexF64) || v"12.2" < CUSPARSE.version())
                        @testset "ldiv! -- (CuVector, CuVector)" begin
                            z  = rand(elty, m)
                            dz = CuArray(z)
                            ldiv!(z, triangle(opa(A)), y)
                            ldiv!(dz, triangle(opa(dA)), dy)
                            @test z ≈ collect(dz)
                        end
                        @testset "\\ -- CuVector" begin
                            x  = triangle(opa(A)) \ y
                            dx = triangle(opa(dA)) \ dy
                            @test x ≈ collect(dx)
                        end
                    end
                    @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                        elty <: Complex && opb == adjoint && continue
                        B  = opb == identity ? rand(elty, m, nB) : rand(elty, nB, m)
                        dB = CuArray(B)
                        B_bad = opb == identity ? rand(elty, m+1, nB) : rand(elty, nB, m+1)
                        dB_bad = CuArray(B_bad)
                        error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                        @testset "ldiv! -- CuMatrix" begin
                            D  = copy(B)
                            dD = copy(dB)
                            ldiv!(triangle(opa(A)), opb(D))
                            ldiv!(triangle(opa(dA)), opb(dD))
                            @test B ≈ collect(dB)
                        end
                        if SparseMatrixType != CuSparseMatrixBSR && (elty ∈ (Float64, ComplexF64) || v"12.2" < CUSPARSE.version())
                            @testset "ldiv! -- (CuMatrix, CuMatrix)" begin
                                C = rand(elty, m, nB)
                                dC = CuArray(C)
                                ldiv!(C, triangle(opa(A)), opb(B))
                                ldiv!(dC, triangle(opa(dA)), opb(dB))
                                @test C ≈ collect(dC)
                            end
                            @testset "\\ -- CuMatrix" begin
                                C  = triangle(opa(A)) \ opb(B)
                                dC = triangle(opa(dA)) \ opb(dB)
                                @test C ≈ collect(dC)
                            end
                        end
                    end
                end
            end
        end
        @testset "CuSparseMatrixCSR($f) $elty" for f in [transpose, adjoint]
            S = f(sprand(elty, m, m, 0.1))
            @test SparseMatrixCSC(CuSparseMatrixCSR(S)) ≈ S

            S = sprand(elty, m, m, 0.1)
            T = f(CuSparseMatrixCSR(S))
            @test SparseMatrixCSC(CuSparseMatrixCSC(T)) ≈ f(S)

            S = sprand(elty, m, m, 0.1)
            T = f(CuSparseMatrixCSC(S))
            @test SparseMatrixCSC(CuSparseMatrixCSR(T)) ≈ f(S)
        end
        mU = 100
        AU = sprand(elty, mU, mU, 0.1)
        U1 = 2*I
        U2 = 2*I(mU)
        U3 = Diagonal(rand(elty, mU))
        for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
            BU = SparseMatrixType(AU)
            for op in (+, -, *)
                @testset "UniformScaling basic operations $op" begin
                    @test Array(op(BU, U1)) ≈ op(AU, U1) && Array(op(U1, BU)) ≈ op(U1, AU)
                end
                @testset "Diagonal basic operations" begin
                    @test Array(op(BU, U2)) ≈ op(AU, U2) && Array(op(U2, BU)) ≈ op(U2, AU)
                    @test Array(op(BU, U3)) ≈ op(AU, U3) && Array(op(U3, BU)) ≈ op(U3, AU)
                end
            end
        end

        @testset "dot(CuVector, CuSparseVector) and dot(CuSparseVector, CuVector) $elty" begin
            x = sprand(elty, n, 0.5)
            y = rand(elty, n)

            dx = CuSparseVector(x)
            dy = CuVector(y)

            @test dot(dx,dy) ≈ dot(x,y)
            @test dot(dy,dx) ≈ dot(y,x)
        end
    end

    @testset "SparseArrays" begin
        @testset "spdiagm(CuVector{$elty})" begin
            ref_vec = collect(elty, 100:121)
            cuda_vec = CuVector(ref_vec)

            ref_spdiagm = spdiagm(ref_vec) # SparseArrays
            cuda_spdiagm = spdiagm(cuda_vec)

            ref_cuda_sparse = CuSparseMatrixCSC(ref_spdiagm)

            @test ref_cuda_sparse.rowVal == cuda_spdiagm.rowVal

            @test ref_cuda_sparse.nzVal == cuda_spdiagm.nzVal

            @test ref_cuda_sparse.colPtr == cuda_spdiagm.colPtr
        end

        @testset "spdiagm(2 => CuVector{$elty})" begin
            ref_vec = collect(elty, 100:121)
            cuda_vec = CuVector(ref_vec)

            ref_spdiagm = spdiagm(2 => ref_vec) # SparseArrays
            cuda_spdiagm = spdiagm(2 => cuda_vec)

            ref_cuda_sparse = CuSparseMatrixCSC(ref_spdiagm)

            @test ref_cuda_sparse.rowVal == cuda_spdiagm.rowVal

            @test ref_cuda_sparse.nzVal == cuda_spdiagm.nzVal

            @test ref_cuda_sparse.colPtr == cuda_spdiagm.colPtr
        end
    end
end
