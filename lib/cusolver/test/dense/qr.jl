using cuSOLVER
using LinearAlgebra

m = 15
n = 10
l = 13

@testset "qr elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    tol = min(m, n)*eps(real(elty))*(1 + (elty <: Complex))

    A              = rand(elty, m, n)
    F              = qr(A)

    d_A            = CuArray(A)
    d_F            = qr(d_A)

    d_RR           = d_F.Q'*d_A
    @test collect(d_RR[1:n,:]) ≈ collect(d_F.R) atol=tol*norm(A)
    @test norm(d_RR[n+1:end,:]) < tol*norm(A)

    d_RRt          = d_A'*d_F.Q
    @test collect(d_RRt[:,1:n]) ≈ collect(d_F.R') atol=tol*norm(A)
    @test norm(d_RRt[:,n+1:end]) < tol*norm(A)

    @test size(d_F) == size(A)
    @test size(d_F.Q) == (m, m)
    @test size(d_F.R) == (n, n)
    @test size(d_RR) == size(d_A)
    @test size(d_RRt) == size(d_A')

    @test CuArray(d_F) ≈ d_A

    d_I = CuMatrix{elty}(I, size(d_F.Q))
    @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A)
    @test collect((d_F.Q'd_I) * d_F.Q) ≈ collect(d_I)
    @test collect(d_F.Q * (d_I * d_F.Q')) ≈ collect(d_I)

    d_I = CuMatrix{elty}(I, size(d_F.R))
    @test collect(d_F.R * d_I) ≈ collect(d_F.R)
    @test collect(d_I * d_F.R) ≈ collect(d_F.R)

    CUDACore.@allowscalar begin
        qval = d_F.Q[1, 1]
        @test qval ≈ F.Q[1, 1]
        qrstr = sprint(show, MIME"text/plain"(), d_F)
        @test qrstr == "$(typeof(d_F))\nQ factor: $(sprint(show, MIME"text/plain"(), d_F.Q))\nR factor:\n$(sprint(show, MIME"text/plain"(), d_F.R))"
    end

    Q, R = F
    dQ, dR = d_F
    @test collect(dQ*dR) ≈ A
    @test collect(dR * dQ') ≈ (R * Q')

    A              = rand(elty, n, m)
    d_A            = CuArray(A)
    d_F            = qr(d_A)
    @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A)

    A              = rand(elty, m, n)
    d_A            = CuArray(A)
    d_q, d_r       = qr(d_A)
    q, r           = qr(A)
    @test Array(d_q) ≈ Array(q)
    @test collect(CuArray{elty}(d_q)) ≈ Array(q)
    @test collect(CuArray(d_q)) ≈ Array(q)
    @test Array(d_r) ≈ Array(r)
    @test CuArray(d_q) ≈ convert(typeof(d_A), d_q)

    A              = rand(elty, n, m)
    d_A            = CuArray(A)
    d_q, d_r       = qr(d_A)
    q, r           = qr(A)
    @test Array(d_q) ≈ Array(q)
    @test Array(d_r) ≈ Array(r)

    A              = rand(elty, n)  # A and B are vectors
    d_A            = CuArray(A)
    M              = qr(A)
    d_M            = qr(d_A)
    B              = rand(elty, n)
    d_B            = CuArray(B)
    @test collect(d_M \ d_B) ≈ M \ B
    @test_throws DimensionMismatch("arguments must have the same number of rows") d_M \ CUDACore.ones(elty, n+1)

    A              = rand(elty, m, n)  # A is a matrix and B,C is a vector
    d_A            = CuArray(A)
    M              = qr(A)
    d_M            = qr(d_A)
    B              = rand(elty, m)
    d_B            = CuArray(B)
    C              = rand(elty, n)
    d_C            = CuArray(C)
    @test collect(d_M \ d_B) ≈ M \ B
    @test collect(d_M.Q * d_B) ≈ (M.Q * B)
    @test collect(d_M.Q' * d_B) ≈ (M.Q' * B)
    @test collect(d_B' * d_M.Q) ≈ (B' * M.Q)
    @test collect(d_B' * d_M.Q') ≈ (B' * M.Q')
    @test collect(d_M.R * d_C) ≈ (M.R * C)
    @test collect(d_M.R' * d_C) ≈ (M.R' * C)
    @test collect(d_C' * d_M.R) ≈ (C' * M.R)
    @test collect(d_C' * d_M.R') ≈ (C' * M.R')

    A              = rand(elty, m, n)  # A and B,C are matrices
    d_A            = CuArray(A)
    M              = qr(A)
    d_M            = qr(d_A)
    B              = rand(elty, m, l) # different second dimension to verify whether dimensions agree
    d_B            = CuArray(B)
    C              = rand(elty, n, l) # different second dimension to verify whether dimensions agree
    d_C            = CuArray(C)
    @test collect(d_M \ d_B) ≈ (M \ B)
    @test collect(d_M.Q * d_B) ≈ (M.Q * B)
    @test collect(d_M.Q' * d_B) ≈ (M.Q' * B)
    @test collect(d_B' * d_M.Q) ≈ (B' * M.Q)
    @test collect(d_B' * d_M.Q') ≈ (B' * M.Q')
    @test collect(d_M.R * d_C) ≈ (M.R * C)
    @test collect(d_M.R' * d_C) ≈ (M.R' * C)
    @test collect(d_C' * d_M.R) ≈ (C' * M.R)
    @test collect(d_C' * d_M.R') ≈ (C' * M.R')
end

@testset "ldiv! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @test testf(rand(elty, m, m), rand(elty, m)) do A, x
        ldiv!(qr(A), x)
        x
    end

    @test testf(rand(elty, m, m), rand(elty, m), rand(elty, m)) do A, x, y
        ldiv!(y, qr(A), x)
        y
    end
end
