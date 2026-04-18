using cuSOLVER
using LinearAlgebra

m = 15
n = 10

@testset "gesvd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, n)
    d_A = CuMatrix(A)
    U, Σ, Vt = cuSOLVER.Xgesvd!('A', 'A', d_A)
    @test A ≈ collect(U[:,1:n] * Diagonal(Σ) * Vt)

    for jobu in ('A', 'S', 'N', 'O')
        for jobvt in ('A', 'S', 'N', 'O')
            (jobu == 'A') && (jobvt == 'A') && continue
            (jobu == 'O') && (jobvt == 'O') && continue
            d_A = CuMatrix(A)
            U2, Σ2, Vt2 = cuSOLVER.Xgesvd!(jobu, jobvt, d_A)
            @test Σ ≈ Σ2
        end
    end
end

@testset "gesvdp! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # nrows > ncols
    A = rand(elty, m, n)
    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.Xgesvdp!('V', 0, d_A)
    @test A ≈ collect(U[:,1:n]) * Diagonal(collect(Σ)) * collect(V)'

    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.Xgesvdp!('V', 1, d_A)
    @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V)'

    d_A = CuMatrix(A)
    Σ2, err_sigma = cuSOLVER.Xgesvdp!('N', 0, d_A)
    @test collect(Σ) ≈ collect(Σ2)

    d_A = CuMatrix(A)
    Σ3, err_sigma = cuSOLVER.Xgesvdp!('N', 1, d_A)
    @test collect(Σ) ≈ collect(Σ3)

    # nrows < ncols
    A = rand(elty, n, m)
    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.Xgesvdp!('V', 0, d_A)
    @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V[:,1:n])'

    d_A = CuMatrix(A)
    U, Σ, V, err_sigma = cuSOLVER.Xgesvdp!('V', 1, d_A)
    @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V)'

    d_A = CuMatrix(A)
    Σ2, err_sigma = cuSOLVER.Xgesvdp!('N', 0, d_A)
    @test collect(Σ) ≈ collect(Σ2)

    d_A = CuMatrix(A)
    Σ3, err_sigma = cuSOLVER.Xgesvdp!('N', 1, d_A)
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

    d_U, d_Σ, d_V = cuSOLVER.Xgesvdr!('N', 'N', d_A, 3)
    @test norm(diag(Σ)[1:3] - collect(d_Σ[1:3])) ≤ tol

    d_U, d_Σ, d_V = cuSOLVER.Xgesvdr!('S', 'S', d_A, ℓ)
    @test norm(diag(Σ) - collect(d_Σ)) ≤ tol
end
