using cuSOLVER
using LinearAlgebra

m = 15
n = 10

if cuSOLVER.version() >= v"11"
@testset "gesvdp! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # nrows > ncols
    A = rand(elty, m, n)
    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.gesvdp!('V', false, d_A)
    @test A ≈ collect(U[:,1:n]) * Diagonal(collect(Σ)) * collect(V)'

    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.gesvdp!('V', true, d_A)
    @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V)'

    d_A = CuMatrix(A)
    Σ2, err_sigma = cuSOLVER.gesvdp!('N', false, d_A)
    @test collect(Σ) ≈ collect(Σ2)

    d_A = CuMatrix(A)
    Σ3, err_sigma = cuSOLVER.gesvdp!('N', true, d_A)
    @test collect(Σ) ≈ collect(Σ3)

    # nrows < ncols
    A = rand(elty, n, m)
    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.gesvdp!('V', false, d_A)
    @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V[:,1:n])'

    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.gesvdp!('V', true, d_A)
    @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V)'

    d_A = CuMatrix(A)
    Σ2, err_sigma = cuSOLVER.gesvdp!('N', false, d_A)
    @test collect(Σ) ≈ collect(Σ2)

    d_A = CuMatrix(A)
    Σ3, err_sigma = cuSOLVER.gesvdp!('N', true, d_A)
    @test collect(Σ) ≈ collect(Σ3)
end

@testset "gesvdr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    R = real(elty)
    tol = R == Float32 ? 1e-2 : 1e-5
    ℓ = min(m, n)

    B = rand(elty, m, m)
    FB = qr(B)
    C = rand(elty, n, n)
    FC = qr(C)
    Σ = zeros(R, m, n)
    for i = 1:ℓ
        Σ[i,i] = (10-i+1)*one(R)
    end
    A = FB.Q * Σ * FC.Q
    d_A = CuMatrix(A)

    d_U, d_Σ, d_V = cuSOLVER.gesvdr!('N', 'N', d_A, 3)
    @test norm(diag(Σ)[1:3] - collect(d_Σ[1:3])) ≤ tol

    d_U, d_Σ, d_V = cuSOLVER.gesvdr!('S', 'S', d_A, ℓ)
    @test norm(diag(Σ) - collect(d_Σ)) ≤ tol
end

end
