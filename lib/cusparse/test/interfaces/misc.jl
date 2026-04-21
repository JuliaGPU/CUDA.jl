using CUDACore, cuSPARSE
using LinearAlgebra, SparseArrays

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

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

    @testset "UniformScaling and Diagonal operations" begin
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
    end

    @testset "dot(CuVector, CuSparseVector) and dot(CuSparseVector, CuVector) $elty" begin
        n = 10
        x = sprand(elty, n, 0.5)
        y = rand(elty, n)

        dx = CuSparseVector(x)
        dy = CuVector(y)

        @test dot(dx,dy) ≈ dot(x,y)
        @test dot(dy,dx) ≈ dot(y,x)
    end
end

@testset "SparseArrays" begin
    @testset "spdiagm(CuVector{$elty})" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        ref_vec = collect(elty, 100:121)
        cuda_vec = CuVector(ref_vec)

        ref_spdiagm = spdiagm(ref_vec) # SparseArrays
        cuda_spdiagm = spdiagm(cuda_vec)

        ref_cuda_sparse = CuSparseMatrixCSC(ref_spdiagm)

        @test ref_cuda_sparse.rowVal == cuda_spdiagm.rowVal
        @test ref_cuda_sparse.nzVal == cuda_spdiagm.nzVal
        @test ref_cuda_sparse.colPtr == cuda_spdiagm.colPtr
    end

    @testset "spdiagm(2 => CuVector{$elty})" for elty in [Float32, Float64, ComplexF32, ComplexF64]
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

@testset "getindex with boolean masks $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 10
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
    CUDACore.@allowscalar for i in eachindex(A, A_csr)
        @test A[i] ≈ A_csr[i]
    end
    S_csr = A_csr[rowmask_d, colmask_d]
    @test S_csr isa CuSparseMatrixCSR
    CUDACore.@allowscalar for i in eachindex(S_cpu, S_csr)
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

    # empty mask (all zeros)
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

    # all-ones mask
    S_all_csc = A_csc[CuVector(trues(m)), CuVector(trues(n))]
    @test S_all_csc isa CuSparseMatrixCSC
    @test collect(S_all_csc) ≈ Matrix(A)

    S_all_csr = A_csr[CuVector(trues(m)), CuVector(trues(n))]
    @test S_all_csr isa CuSparseMatrixCSR
    CUDACore.@allowscalar for i in eachindex(A, S_all_csr)
        @test A[i] ≈ S_all_csr[i]
    end

    # zero-dimension matrix
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
