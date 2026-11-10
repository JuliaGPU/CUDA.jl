using cuSOLVER
using LinearAlgebra

n = 8

if cuSOLVER.version() >= v"11"
@testset "deprecated names, elty = $elty" for elty in [Float32, ComplexF64]
    A = rand(elty, n, n)
    S = A + A'
    P = A * A' + I
    B = rand(elty, n, 2)

    d_A, d_ipiv, _ = @test_deprecated cuSOLVER.Xgetrf!(CuMatrix(A))
    @test d_ipiv isa CuVector{Int64}
    d_A, d_ipiv, _ = @test_deprecated cuSOLVER.Xgetrf!(d_A, d_ipiv)
    d_B = @test_deprecated cuSOLVER.Xgetrs!('N', cuSOLVER.getrf!(CuMatrix(A), d_ipiv)[1], d_ipiv, CuMatrix(B))
    @test collect(d_B) ≈ A \ B

    d_P, _ = @test_deprecated cuSOLVER.Xpotrf!('U', CuMatrix(P))
    d_B = @test_deprecated cuSOLVER.Xpotrs!('U', d_P, CuMatrix(B))
    @test collect(d_B) ≈ P \ B

    d_Q, tau = @test_deprecated cuSOLVER.Xgeqrf!(CuMatrix(A))
    @test_deprecated cuSOLVER.Xgeqrf!(CuMatrix(A), tau)

    U, Σ, Vt = @test_deprecated cuSOLVER.Xgesvd!('S', 'S', CuMatrix(A))
    @test collect(Σ) ≈ svdvals(A)
    Σ, _ = @test_deprecated cuSOLVER.Xgesvdp!('N', 1, CuMatrix(A))
    @test collect(Σ) ≈ svdvals(A)
    Σ, _ = @test_deprecated cuSOLVER.Xgesvdp!('N', true, CuMatrix(A))
    @test collect(Σ) ≈ svdvals(A)
    _, Σ, _ = @test_deprecated cuSOLVER.Xgesvdr!('N', 'N', CuMatrix(A), 2)
    @test length(Σ) == n

    W = @test_deprecated cuSOLVER.Xsyevd!('N', 'U', CuMatrix(S))
    @test collect(W) ≈ eigvals(Hermitian(S))
    W, _ = @test_deprecated cuSOLVER.Xsyevdx!('N', 'A', 'U', CuMatrix(S))
    @test collect(W) ≈ eigvals(Hermitian(S))
    if elty <: Complex
        W = @test_deprecated cuSOLVER.heevd!('N', 'U', CuMatrix(S))
        @test collect(W) ≈ eigvals(Hermitian(S))
    end

    if cuSOLVER.version() >= v"11.7.1"
        W, _, _ = @test_deprecated cuSOLVER.Xgeev!('N', 'N', CuMatrix(A))
        @test sort(abs.(collect(W))) ≈ sort(abs.(eigvals(A)))
        W = @test_deprecated cuSOLVER.XsyevBatched!('N', 'U', CuArray(cat(S, S; dims=3)))
        @test collect(W) ≈ repeat(eigvals(Hermitian(S)), 1, 2)
    end
end
end
