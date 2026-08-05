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
    @test LAPACK.geqrf!(CuArray(A))[2] ≈ τ
    dB, τ_b = LAPACK.geqrf!(CuArray(A), similar(τ))
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

            # blocked application of the reflectors (used for very tall matrices)
            for blocksize in (1, 3, 4)
                dC = CuArray(C)
                cuSOLVER.ormqr!(side, trans, dA, dτ, dC; blocksize)
                @test dC ≈ dD
                dc = CuArray(C[:, 1])
                if side == 'L'
                    cuSOLVER.ormqr!(side, trans, dA, dτ, dc; blocksize)
                    @test dc ≈ dD[:, 1]
                end
            end
        end
    end
end

# the blocked fallback for problems that cuSOLVER's 32-bit workspace size cannot describe,
# compared against LAPACK for arbitrary reflectors, including some with τ = 0
@testset "blocked ormqr! and orgqr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    trans_types = elty <: Real ? ('N', 'T', 'C') : ('N', 'C')
    @testset "size = $((r, c))" for (r, c) in ((m, n), (n, m), (n, n))
        k = min(r, c)
        A = rand(elty, r, c)
        τ = rand(elty, k)
        τ[[1, k ÷ 2]] .= 0
        dA, dτ = CuArray(A), CuArray(τ)
        @testset "blocksize = $blocksize" for blocksize in (1, 3, 64)
            @testset "trans = $trans" for trans in trans_types
                # LAPACK only accepts 'C' for complex matrices
                ref_trans = elty <: Real && trans == 'C' ? 'T' : trans

                B = rand(elty, r, 4)
                dB = CuArray(B)
                cuSOLVER.ormqr!('L', trans, dA, dτ, dB; blocksize)
                @test Array(dB) ≈ LAPACK.ormqr!('L', ref_trans, copy(A), τ, copy(B))

                B = rand(elty, 4, r)
                dB = CuArray(B)
                cuSOLVER.ormqr!('R', trans, dA, dτ, dB; blocksize)
                @test Array(dB) ≈ LAPACK.ormqr!('R', ref_trans, copy(A), τ, copy(B))

                b = rand(elty, r)
                db = CuArray(b)
                cuSOLVER.ormqr!('L', trans, dA, dτ, db; blocksize)
                @test Array(db) ≈ LAPACK.ormqr!('L', ref_trans, copy(A), τ, copy(b))

                db = CuArray([b; 0])
                cuSOLVER.ormqr!('L', trans, dA, dτ, view(db, 1:1:r); blocksize)
                @test Array(db)[1:r] ≈ LAPACK.ormqr!('L', ref_trans, copy(A), τ, copy(b))
            end

            Q = LAPACK.orgqr!(copy(A), τ)[:, 1:k]
            @test Array(cuSOLVER.orgqr!(copy(dA), dτ; blocksize)) ≈ Q
            # fewer reflectors than columns of Q
            @test Array(cuSOLVER.orgqr!(copy(dA), dτ[1:k-2]; blocksize)) ≈
                  LAPACK.orgqr!(copy(A), τ[1:k-2])[:, 1:k]
        end
    end

    # strided inputs and outputs
    A = rand(elty, m + 2, n + 1)
    τ = rand(elty, n)
    dA, dτ = CuArray(A), CuArray(τ)
    B = rand(elty, m + 3, 5)
    dB = CuArray(B)
    cuSOLVER.ormqr!('L', 'N', view(dA, 2:m+1, 2:n+1), dτ, view(dB, 2:m+1, 2:4); blocksize=3)
    B[2:m+1, 2:4] = LAPACK.ormqr!('L', 'N', A[2:m+1, 2:n+1], τ, B[2:m+1, 2:4])
    @test Array(dB) ≈ B
    cuSOLVER.orgqr!(view(dA, 2:m+1, 2:n+1), dτ; blocksize=3)
    A[2:m+1, 2:n+1] = LAPACK.orgqr!(A[2:m+1, 2:n+1], τ)
    @test Array(dA) ≈ A
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
    @test LAPACK.getrf!(CuArray(A))[1] ≈ d_A
    @test LAPACK.getrf!(CuArray(A), similar(d_ipiv))[1] ≈ d_A

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
    @test LAPACK.getrs!('N', d_A, d_ipiv, CuArray(B)) ≈ d_B
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
    @test LAPACK.sytrf!('U', d_B)[1] ≈ d_A
    @test LAPACK.sytrf!('U', d_C, similar(d_ipiv))[1] ≈ d_A
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
    @test LAPACK.gebrd!(d_B)[1] ≈ d_A
end
