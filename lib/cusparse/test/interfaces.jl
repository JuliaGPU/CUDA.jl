using cuSPARSE
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
                    if SparseMatrixType != CuSparseMatrixBSR && (elty ∈ (Float64, ComplexF64) || v"12.2" < cuSPARSE.version())
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
                        if SparseMatrixType != CuSparseMatrixBSR && (elty ∈ (Float64, ComplexF64) || v"12.2" < cuSPARSE.version())
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

    @testset "getindex with boolean masks" begin
        A = sprand(elty, m, n, 0.4)
        rowmask = rand(Bool, m)
        colmask = rand(Bool, n)
        S_cpu = A[rowmask, colmask]

        rowmask_d = CuVector(rowmask)
        colmask_d = CuVector(colmask)

        # test slicing of CSC format
        A_csc = CuSparseMatrixCSC(A)
        S_csc = A_csc[rowmask_d, colmask_d]
        @test S_csc isa CuSparseMatrixCSC
        @test S_cpu ≈ collect(S_csc)

        # test slicing of CSR format
        # Conversion between CSC and CSR is broken in many ways on CUDA 12.0,
        # therefore we construct the CSR matrix manually from the transposed CSC.
        Aᵀ_csc = CuSparseMatrixCSC(Transpose(A))
        A_csr = CuSparseMatrixCSR{eltype(A), Int32}(
            copy(Aᵀ_csc.colPtr), # rowPtr is the same as colPtr of the transposed CSC
            copy(Aᵀ_csc.rowVal), # colVal is the same as rowVal of the transposed CSC
            copy(Aᵀ_csc.nzVal),  # nzVal is unchanged by transposition
            size(A)
        )
        # collect calls CSR→CSC conversion again which is broken, so we test on scalar level
        CUDA.@allowscalar for i in eachindex(A, A_csr)
            @test A[i] ≈ A_csr[i]
        end
        S_csr = A_csr[rowmask_d, colmask_d]
        @test S_csr isa CuSparseMatrixCSR
        CUDA.@allowscalar for i in eachindex(S_cpu, S_csr)
            @test S_cpu[i] ≈ S_csr[i]
        end

        # wrong mask size: throws BoundsError for both too-long and too-short, matching the behaviour of dense Array.
        @test_throws BoundsError A_csc[CuVector(trues(m + 1)), colmask_d]
        @test_throws BoundsError A_csc[rowmask_d, CuVector(trues(n + 1))]
        @test_throws BoundsError A_csc[CuVector(trues(m - 1)), colmask_d]
        @test_throws BoundsError A_csc[rowmask_d, CuVector(trues(n - 1))]
        @test_throws BoundsError A_csr[CuVector(trues(m + 1)), colmask_d]
        @test_throws BoundsError A_csr[rowmask_d, CuVector(trues(n + 1))]
        @test_throws BoundsError A_csr[CuVector(trues(m - 1)), colmask_d]
        @test_throws BoundsError A_csr[rowmask_d, CuVector(trues(n - 1))]

        # empty mask (all zeros): cumsum gives all-zero rowmap/colmap, new_m or new_n = 0,
        # both kernels are guarded by `new_m > 0 && new_n > 0`, so nothing executes.
        # new_rowPtr collapses to [1] (or all-ones), nnz = 0. Same as CPU SparseArrays.
        S_empty_rows_csc = A_csc[CuVector(falses(m)), CuVector(trues(n))]
        @test S_empty_rows_csc isa CuSparseMatrixCSC
        @test size(S_empty_rows_csc) == (0, n)
        @test nnz(S_empty_rows_csc) == 0

        S_empty_cols_csc = A_csc[CuVector(trues(m)), CuVector(falses(n))]
        @test S_empty_cols_csc isa CuSparseMatrixCSC
        @test size(S_empty_cols_csc) == (m, 0)
        @test nnz(S_empty_cols_csc) == 0

        S_empty_rows_csr = A_csr[CuVector(falses(m)), CuVector(trues(n))]
        @test S_empty_rows_csr isa CuSparseMatrixCSR
        @test size(S_empty_rows_csr) == (0, n)
        @test nnz(S_empty_rows_csr) == 0

        S_empty_cols_csr = A_csr[CuVector(trues(m)), CuVector(falses(n))]
        @test S_empty_cols_csr isa CuSparseMatrixCSR
        @test size(S_empty_cols_csr) == (m, 0)
        @test nnz(S_empty_cols_csr) == 0

        S_empty_both_csc = A_csc[CuVector(falses(m)), CuVector(falses(n))]
        @test S_empty_both_csc isa CuSparseMatrixCSC
        @test size(S_empty_both_csc) == (0, 0)
        @test nnz(S_empty_both_csc) == 0

        S_empty_both_csr = A_csr[CuVector(falses(m)), CuVector(falses(n))]
        @test S_empty_both_csr isa CuSparseMatrixCSR
        @test size(S_empty_both_csr) == (0, 0)
        @test nnz(S_empty_both_csr) == 0

        # all-ones mask: rowmap = 1:m, colmap = 1:n, both kernels run unfiltered.
        # Result should equal the full matrix. Same as CPU SparseArrays.
        S_all_csc = A_csc[CuVector(trues(m)), CuVector(trues(n))]
        @test S_all_csc isa CuSparseMatrixCSC
        @test collect(S_all_csc) ≈ Matrix(A)

        S_all_csr = A_csr[CuVector(trues(m)), CuVector(trues(n))]
        @test S_all_csr isa CuSparseMatrixCSR
        CUDA.@allowscalar for i in eachindex(A, S_all_csr)
            @test A[i] ≈ S_all_csr[i]
        end

        # zero-dimension matrix: accessing rowmap[end] / colmap[end] on an empty CuVector
        # would crash without the `m > 0` / `n > 0` guard in the implementation.
        A_zero_rows_csr = CuSparseMatrixCSR(spzeros(elty, 0, n))
        S_zr = A_zero_rows_csr[CuVector{Bool}([]), CuVector(trues(n))]
        @test S_zr isa CuSparseMatrixCSR
        @test size(S_zr) == (0, n)
        @test nnz(S_zr) == 0

        A_zero_cols_csr = CuSparseMatrixCSR(spzeros(elty, m, 0))
        S_zc = A_zero_cols_csr[CuVector(trues(m)), CuVector{Bool}([])]
        @test S_zc isa CuSparseMatrixCSR
        @test size(S_zc) == (m, 0)
        @test nnz(S_zc) == 0

        A_zero_both_csr = CuSparseMatrixCSR(spzeros(elty, 0, 0))
        S_zb = A_zero_both_csr[CuVector{Bool}([]), CuVector{Bool}([])]
        @test S_zb isa CuSparseMatrixCSR
        @test size(S_zb) == (0, 0)
        @test nnz(S_zb) == 0

        A_zero_rows_csc = CuSparseMatrixCSC(spzeros(elty, 0, n))
        S_zr_csc = A_zero_rows_csc[CuVector{Bool}([]), CuVector(trues(n))]
        @test S_zr_csc isa CuSparseMatrixCSC
        @test size(S_zr_csc) == (0, n)
        @test nnz(S_zr_csc) == 0

        A_zero_cols_csc = CuSparseMatrixCSC(spzeros(elty, m, 0))
        S_zc_csc = A_zero_cols_csc[CuVector(trues(m)), CuVector{Bool}([])]
        @test S_zc_csc isa CuSparseMatrixCSC
        @test size(S_zc_csc) == (m, 0)
        @test nnz(S_zc_csc) == 0

        A_zero_both_csc = CuSparseMatrixCSC(spzeros(elty, 0, 0))
        S_zb_csc = A_zero_both_csc[CuVector{Bool}([]), CuVector{Bool}([])]
        @test S_zb_csc isa CuSparseMatrixCSC
        @test size(S_zb_csc) == (0, 0)
        @test nnz(S_zb_csc) == 0
    end
end
