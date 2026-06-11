using cuSOLVER
using LinearAlgebra

m = 15
n = 10

@testset "Cholesky (po) elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A    = rand(elty, n, n)
    A    = A*A'+I #posdef
    B    = rand(elty, n, n)
    d_A  = CuArray(A)
    d_B  = CuArray(B)

    pivot = NoPivot()
    d_F  = cholesky(d_A, pivot)
    F    = cholesky(A, pivot)
    @test F.U   ≈ collect(d_F.U)
    @test F\(A'B) ≈ collect(d_F\(d_A'd_B))

    d_F  = cholesky(Hermitian(d_A, :L), pivot)
    F    = cholesky(Hermitian(A, :L), pivot)
    @test F.L   ≈ collect(d_F.L)
    @test F\(A'B) ≈ collect(d_F\(d_A'd_B))

    @test_throws DimensionMismatch LinearAlgebra.LAPACK.potrs!('U', d_A, CuArray(rand(elty, m, m)))

    A    = rand(elty, m, n)
    d_A  = CuArray(A)
    @test_throws DimensionMismatch cholesky(d_A)
    @test_throws DimensionMismatch LinearAlgebra.LAPACK.potrs!('U', d_A, d_B)

    A    = zeros(elty, n, n)
    d_A  = CuArray(A)
    @test_throws LinearAlgebra.PosDefException cholesky(d_A)
end

@testset "Cholesky inverse (potri) elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # test lower
    A    = rand(elty, n, n)
    A    = A*A'+I #posdef
    d_A  = CuArray(A)

    LinearAlgebra.LAPACK.potrf!('L', A)
    LinearAlgebra.LAPACK.potrf!('L', d_A)

    LinearAlgebra.LAPACK.potri!('L', A)
    LinearAlgebra.LAPACK.potri!('L', d_A)
    @test A  ≈ collect(d_A)

    # test upper
    A    = rand(elty, n, n)
    A    = A*A'+I #posdef
    d_A  = CuArray(A)

    LinearAlgebra.LAPACK.potrf!('U', A)
    LinearAlgebra.LAPACK.potrf!('U', d_A)
    LinearAlgebra.LAPACK.potri!('U', A)
    LinearAlgebra.LAPACK.potri!('U', d_A)
    @test A  ≈ collect(d_A)
end

@testset "potrsBatched! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # Test lower
    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]
    bB = [rand(elty, m) for i in 1:n]

    bd_A = CuArray{elty, 2}[]
    bd_B = CuArray{elty, 1}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
        push!(bd_B, CuArray(bB[i]))
    end

    bd_X = cuSOLVER.potrsBatched!('L', bd_A, bd_B)
    bh_X = [collect(bd_X[i]) for i in 1:n]

    for i = 1:n
        LinearAlgebra.LAPACK.potrs!('L', bA[i], bB[i])
        @test bB[i] ≈ bh_X[i]
    end

    # Test upper
    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]
    bB = [rand(elty, m) for i in 1:n]

    bd_A = CuArray{elty, 2}[]
    bd_B = CuArray{elty, 1}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
        push!(bd_B, CuArray(bB[i]))
    end

    bd_X = cuSOLVER.potrsBatched!('U', bd_A, bd_B)
    bh_X = [collect(bd_X[i]) for i in 1:n]

    for i = 1:n
        LinearAlgebra.LAPACK.potrs!('U', bA[i], bB[i])
        @test bB[i] ≈ bh_X[i]
    end

    # error throwing tests
    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]
    bB = [rand(elty, m) for i in 1:n+1]

    bd_A = CuArray{elty, 2}[]
    bd_B = CuArray{elty, 1}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
        push!(bd_B, CuArray(bB[i]))
    end
    push!(bd_B, CuArray(bB[end]))

    @test_throws DimensionMismatch cuSOLVER.potrsBatched!('L', bd_A, bd_B)

    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]
    bB = [rand(elty, m) for i in 1:n]
    bB[1] = rand(elty, m+1)
    bd_A = CuArray{elty, 2}[]
    bd_B = CuArray{elty, 1}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
        push!(bd_B, CuArray(bB[i]))
    end

    @test_throws DimensionMismatch cuSOLVER.potrsBatched!('L', bd_A, bd_B)

    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]
    bB = [rand(elty, m, m) for i in 1:n]
    bd_A = CuArray{elty, 2}[]
    bd_B = CuArray{elty, 2}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
        push!(bd_B, CuArray(bB[i]))
    end

    @test_throws ArgumentError cuSOLVER.potrsBatched!('L', bd_A, bd_B)
end

@testset "potrfBatched! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # Test lower
    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]

    bd_A = CuArray{elty, 2}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
    end

    bd_A, info = cuSOLVER.potrfBatched!('L', bd_A)
    bh_A = [collect(bd_A[i]) for i in 1:n]

    for i = 1:n
        LinearAlgebra.LAPACK.potrf!('L', bA[i])
        @test bA[i] ≈ bh_A[i]
    end

    # Test upper
    bA = [rand(elty, m, m) for i in 1:n]
    bA = [bA[i]*bA[i]' for i in 1:n]

    bd_A = CuArray{elty, 2}[]
    for i in 1:length(bA)
        push!(bd_A, CuArray(bA[i]))
    end

    bd_A, info = cuSOLVER.potrfBatched!('U', bd_A)
    bh_A = [collect(bd_A[i]) for i in 1:n]

    for i = 1:n
        LinearAlgebra.LAPACK.potrf!('U', bA[i])
        # cuSOLVER seems to return symmetric/hermitian matrix when using 'U'
        @test Hermitian(bA[i]) ≈ bh_A[i]
    end
end
