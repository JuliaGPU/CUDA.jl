T = Float64
A = sprand(T, m, n, 0.3)
S = sprand(T, m, m, 0.3) + I     # square, with a nonzero diagonal
L = tril(S)
x = rand(T, n)
y = rand(T, m)
B = rand(T, m, k)

@testset "mv2!" begin
    d_y = CuArray(y)
    @test_deprecated cuSPARSE.mv2!('N', 2.0, CuSparseMatrixBSR(A, blockdim), CuArray(x), 3.0, d_y, 'O')
    @test collect(d_y) ≈ 2 * A * x + 3 * y
end

@testset "positional index" begin
    sx = sprand(T, n, 0.3)
    d_sx = CuSparseVector(sx)
    d_x = CuArray(x)

    @test_deprecated cuSPARSE.gather!(copy(d_sx), d_x, 'O')
    d_dense = CUDACore.zeros(T, n)
    @test collect(@test_deprecated cuSPARSE.scatter!(d_dense, d_sx, 'O')) ≈ Vector(sx)
    @test collect(@test_deprecated cuSPARSE.axpby!(2.0, d_sx, 3.0, copy(d_x), 'O')) ≈ 2 * sx + 3 * x
    X, Y = @test_deprecated cuSPARSE.rot!(copy(d_sx), copy(d_x), 1.0, 0.0, 'O')
    @test collect(Y) ≈ x
    @test (@test_deprecated cuSPARSE.vv!('N', d_sx, d_x, 'O')) ≈ dot(sx, x)

    d_A = CuSparseMatrixCSR(A)
    d_y = CuArray(y)
    @test collect(@test_deprecated cuSPARSE.mv!('N', 2.0, d_A, d_x, 3.0, copy(d_y), 'O')) ≈ 2 * A * x + 3 * y
    @test collect(@test_deprecated cuSPARSE.mv!('N', 2.0, d_A, d_x, 3.0, copy(d_y), 'O', cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT)) ≈ 2 * A * x + 3 * y

    C = rand(T, n, k)
    @test collect(@test_deprecated cuSPARSE.mm!('N', 'N', 2.0, d_A, CuArray(C), 3.0, CuArray(B), 'O')) ≈ 2 * A * C + 3 * B
    @test collect(@test_deprecated cuSPARSE.mm!('N', 'N', 2.0, CuArray(collect(A')), CuSparseMatrixCSR(sparse(B)), 3.0, CuArray(C), 'O', cuSPARSE.CUSPARSE_SPMM_ALG_DEFAULT)) ≈ 2 * A' * B + 3 * C

    if cuSPARSE.version() >= v"11.7.2"
        d_Ab = cat(CuSparseMatrixCSR(A), CuSparseMatrixCSR(A); dims=3)
        Cb = rand(T, n, k, 2)
        Bb = rand(T, m, k, 2)
        res = @test_deprecated cuSPARSE.bmm!('N', 'N', 1.0, d_Ab, CuArray(Cb), 0.0, CuArray(Bb), 'O')
        @test collect(res)[:, :, 2] ≈ A * Cb[:, :, 2]
    end

    d_S = CuSparseMatrixCSR(S)
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.gemm('N', 'N', 1.0, d_S, d_S, 'O')) ≈ S * S
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.gemm('N', 'N', 1.0, d_S, d_S, 'O', cuSPARSE.CUSPARSE_SPGEMM_DEFAULT)) ≈ S * S
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.gemm('N', 'N', 1.0, d_S, d_S, 2.0, d_S, 'O')) ≈ S * S + 2 * S
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.gemm('N', 'N', 1.0, d_S, d_S, 0.0, d_S, 'O'; same_pattern=false)) ≈ S * S
    d_C = copy(d_S)
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.gemm!('N', 'N', 1.0, d_S, d_S, 0.0, d_C, 'O')) ≈ S * S
    sy = sprand(T, m, 0.5)
    @test Vector(SparseVector(@test_deprecated cuSPARSE.gemv('N', 1.0, d_S, CuSparseVector(sy), 'O'))) ≈ S * sy

    if cuSPARSE.version() >= v"11.5"
        res = @test_deprecated cuSPARSE.sv!('N', 'L', 'N', 1.0, CuSparseMatrixCSR(L), CuArray(y), CUDACore.zeros(T, m), 'O')
        @test collect(res) ≈ L \ y
    end
    if cuSPARSE.version() >= v"11.6"
        res = @test_deprecated cuSPARSE.sm!('N', 'N', 'L', 'N', 1.0, CuSparseMatrixCSR(L), CuArray(B), CUDACore.zeros(T, m, k), 'O')
        @test collect(res) ≈ L \ B
    end
    if cuSPARSE.version() >= v"11.4.1"
        D = rand(T, m, k)
        E = rand(T, k, m)
        res = @test_deprecated cuSPARSE.sddmm!('N', 'N', 1.0, CuArray(D), CuArray(E), 0.0, copy(d_S), 'O')
        @test collect(res) ≈ (D * E) .* (S .!= 0)
    end

    d_L = CuSparseMatrixBSR(CuSparseMatrixCSR(L), blockdim)
    @test collect(@test_deprecated cuSPARSE.sv2!('N', 'L', 'N', 1.0, d_L, CuArray(y), 'O')) ≈ L \ y
    @test collect(@test_deprecated cuSPARSE.sv2('N', 'L', 'N', 1.0, d_L, CuArray(y), 'O')) ≈ L \ y
    @test collect(@test_deprecated cuSPARSE.sv2('N', 'L', 'N', d_L, CuArray(y), 'O')) ≈ L \ y
    @test collect(@test_deprecated cuSPARSE.sv2('N', 'L', d_L, CuArray(y), 'O')) ≈ L \ y
    @test collect(@test_deprecated cuSPARSE.sm2!('N', 'N', 'L', 'N', 1.0, d_L, CuArray(B), 'O')) ≈ L \ B
    @test collect(@test_deprecated cuSPARSE.sm2('N', 'N', 'L', 'N', 1.0, d_L, CuArray(B), 'O')) ≈ L \ B
    @test collect(@test_deprecated cuSPARSE.sm2('N', 'N', 'L', 'N', d_L, CuArray(B), 'O')) ≈ L \ B
    @test collect(@test_deprecated cuSPARSE.sm2('N', 'N', 'L', d_L, CuArray(B), 'O')) ≈ L \ B

    D = rand(T, m, n)
    @test collect(@test_deprecated cuSPARSE.gemvi!('N', 2.0, CuArray(D), d_sx, 3.0, copy(d_y), 'O')) ≈ 2 * D * sx + 3 * y

    @test SparseMatrixCSC(@test_deprecated cuSPARSE.geam(2.0, d_S, 3.0, d_S, 'O')) ≈ 5 * S
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.geam(2.0, CuSparseMatrixCSC(S), 3.0, CuSparseMatrixCSC(S), 'O')) ≈ 5 * S
    @test SparseVector(@test_deprecated cuSPARSE.axpby(2.0, d_sx, 3.0, d_sx, 'O')) ≈ 5 * sx

    P = S + S'
    ncolors, coloring, _ = @test_deprecated cuSPARSE.color(CuSparseMatrixCSR(P), 'O')
    @test maximum(collect(coloring)) == ncolors
    ncolors, _, _ = @test_deprecated cuSPARSE.color(CuSparseMatrixCSC(P), 'O'; percentage=0.5)
    @test ncolors >= 1

    H = sparse(Matrix(L) * Matrix(L)') + m * I   # symmetric positive definite
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.ic02(CuSparseMatrixCSR(H), 'O')) ≈ SparseMatrixCSC(cuSPARSE.ic02(CuSparseMatrixCSR(H)))
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.ic02!(CuSparseMatrixCSR(H), 'O')) ≈ SparseMatrixCSC(cuSPARSE.ic02(CuSparseMatrixCSR(H)))
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.ilu02(CuSparseMatrixCSR(S), 'O')) ≈ SparseMatrixCSC(cuSPARSE.ilu02(CuSparseMatrixCSR(S)))
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.ilu02!(CuSparseMatrixCSC(S), 'O')) ≈ SparseMatrixCSC(cuSPARSE.ilu02(CuSparseMatrixCSC(S)))

    @test SparseMatrixCSC(@test_deprecated cuSPARSE.sort_csr(CuSparseMatrixCSR(A), 'O')) ≈ A
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.sort_csc(CuSparseMatrixCSC(A), 'O')) ≈ A
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.prune(CuSparseMatrixCSR(A), 0.5, 'O')) ≈ A .* (abs.(A) .> 0.5)
    @test SparseMatrixCSC(@test_deprecated cuSPARSE.prune(CuSparseMatrixCSC(A), 0.5, 'O')) ≈ A .* (abs.(A) .> 0.5)
end

@testset "gtsv2 index" begin
    dl = rand(T, m); d = rand(T, m) .+ 4; du = rand(T, m)
    dl[1] = 0; du[end] = 0
    M = Tridiagonal(dl[2:end], d, du[1:end-1])
    @test collect(@test_deprecated cuSPARSE.gtsv2!(CuArray(dl), CuArray(d), CuArray(du), CuArray(B), 'O')) ≈ M \ B
    @test collect(@test_deprecated cuSPARSE.gtsv2(CuArray(dl), CuArray(d), CuArray(du), CuArray(B), 'O'; pivoting=false)) ≈ M \ B
end

@testset "sort_coo type" begin
    coo = CuSparseMatrixCOO(CuArray{Cint}([3, 1, 2]), CuArray{Cint}([1, 3, 2]), CuArray([1.0, 2.0, 3.0]), (3, 3))
    sorted = @test_deprecated cuSPARSE.sort_coo(coo, 'R')
    @test collect(sorted.rowInd) == [1, 2, 3]
    sorted = @test_deprecated cuSPARSE.sort_coo(coo, 'C')
    @test collect(sorted.colInd) == [1, 2, 3]
    @test_deprecated @test_throws ArgumentError cuSPARSE.sort_coo(coo, 'X')
end
