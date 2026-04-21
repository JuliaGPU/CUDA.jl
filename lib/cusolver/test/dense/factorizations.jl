using cuSOLVER
using LinearAlgebra
using LinearAlgebra: BlasInt

m = 15
n = 10

@testset "geqrf! -- orgqr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, n)
    dA = CuArray(A)
    dA, τ = cuSOLVER.geqrf!(dA)
    cuSOLVER.orgqr!(dA, τ)
    @test dA' * dA ≈ I
    dB = CuArray(A)
    dB, τ_b = LAPACK.geqrf!(dB)
    LAPACK.orgqr!(dB, τ_b)
    @test dB ≈ dA
    @test τ ≈ τ_b
    dB, τ_b = LAPACK.geqrf!(dB, similar(τ_b))
    LAPACK.orgqr!(dB, τ_b)
    @test dB ≈ dA
end

@testset "ormqr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "side = $side" for side in ['L', 'R']
        @testset "trans = $trans" for (trans, op) in [('N', identity), ('T', transpose), ('C', adjoint)]
            (elty <: Complex) && (trans == 'T') && continue
            A = rand(elty, m, n)
            dA = CuArray(A)
            dA, dτ = cuSOLVER.geqrf!(dA)

            hI = Matrix{elty}(I, m, m)
            dI = CuArray(hI)
            dH = cuSOLVER.ormqr!(side, 'N', dA, dτ, dI)
            @test dH' * dH ≈ I

            C = side == 'L' ? rand(elty, m, n) : rand(elty, n, m)
            dC = CuArray(C)
            dD = side == 'L' ? op(dH) * dC : dC * op(dH)

            cuSOLVER.ormqr!(side, trans, dA, dτ, dC)
            @test dC ≈ dD
        end
    end
end

@testset "getrf! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, n)
    d_A = CuArray(A)
    d_A, d_ipiv, info = cuSOLVER.getrf!(d_A)
    LinearAlgebra.checknonsingular(info)
    h_A = collect(d_A)
    h_ipiv = collect(d_ipiv)
    alu = LinearAlgebra.LU(h_A, convert(Vector{BlasInt}, h_ipiv), zero(BlasInt))
    @test A ≈ Array(alu)
    d_B = CuArray(A)
    d_B, d_ipiv, info = LAPACK.getrf!(d_B)
    @test d_B ≈ d_A
    d_B = CuArray(A)
    d_B, d_ipiv, info = LAPACK.getrf!(d_B, similar(d_ipiv))
    @test d_B ≈ d_A

    d_A, d_ipiv, info = cuSOLVER.getrf!(CUDACore.zeros(elty, n, n))
    @test_throws LinearAlgebra.SingularException LinearAlgebra.checknonsingular(info)
end

@testset "getrs! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A      = rand(elty, n, n)
    d_A    = CuArray(A)
    d_A, d_ipiv = cuSOLVER.getrf!(d_A)
    B      = rand(elty, n, n)
    d_B    = CuArray(B)
    d_B    = cuSOLVER.getrs!('N', d_A, d_ipiv, d_B)
    d_C    = CuArray(B)
    d_C    = LAPACK.getrs!('N', d_A, d_ipiv, d_C)
    @test d_C  ≈ d_B
    h_B    = collect(d_B)
    @test h_B  ≈ A\B
    A      = rand(elty, m, n)
    d_A    = CuArray(A)
    @test_throws DimensionMismatch cuSOLVER.getrs!('N', d_A, d_ipiv, d_B)
    A      = rand(elty, n, n)
    d_A    = CuArray(A)
    B      = rand(elty, m, n)
    d_B    = CuArray(B)
    @test_throws DimensionMismatch cuSOLVER.getrs!('N', d_A, d_ipiv, d_B)
end

@testset "sytrf! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, n, n)
    A = A + A' #symmetric
    d_A = CuArray(A)
    d_B = CuArray(A)
    d_C = CuArray(A)
    d_A, d_ipiv, info = cuSOLVER.sytrf!('U', d_A)
    LinearAlgebra.checknonsingular(info)
    h_A = collect(d_A)
    h_ipiv = collect(d_ipiv)
    A, ipiv = LAPACK.sytrf!('U', A)
    @test ipiv == h_ipiv
    @test A ≈ h_A
    d_B, ipiv_b = LAPACK.sytrf!('U', d_B)
    @test d_B ≈ d_A
    d_C, ipiv_b = LAPACK.sytrf!('U', d_C, similar(ipiv_b))
    @test d_C ≈ d_A
    A    = rand(elty, m, n)
    d_A  = CuArray(A)
    @test_throws DimensionMismatch cuSOLVER.sytrf!('U', d_A)

    d_A, d_ipiv, info = cuSOLVER.sytrf!('U', CUDACore.zeros(elty, n, n))
    @test_throws LinearAlgebra.SingularException LinearAlgebra.checknonsingular(info)
end

@testset "gebrd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A                             = rand(elty, m, n)
    d_A                           = CuArray(A)
    d_B                           = CuArray(A)
    d_A, d_D, d_E, d_TAUQ, d_TAUP = cuSOLVER.gebrd!(d_A)
    h_A                           = collect(d_A)
    h_D                           = collect(d_D)
    h_E                           = collect(d_E)
    h_TAUQ                        = collect(d_TAUQ)
    h_TAUP                        = collect(d_TAUP)
    A, d, e, q, p                 = LAPACK.gebrd!(A)
    @test A ≈ h_A
    @test d ≈ h_D
    @test e[min(m,n)-1] ≈ h_E[min(m,n)-1]
    @test q ≈ h_TAUQ
    @test p ≈ h_TAUP
    d_B, d_D, d_E, d_TAUQ, d_TAUP = LAPACK.gebrd!(d_B)
    @test d_B ≈ d_A
end
