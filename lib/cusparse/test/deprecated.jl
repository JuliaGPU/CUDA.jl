@testset "mv2!" begin
    A = sparse(rand(Float32, m, n))
    x = rand(Float32, n)
    y = rand(Float32, m)
    d_y = CuArray(y)
    @test_deprecated cuSPARSE.mv2!('N', 2f0, CuSparseMatrixBSR(A, blockdim), CuArray(x), 3f0, d_y, 'O')
    @test collect(d_y) ≈ 2 * A * x + 3 * y
end
