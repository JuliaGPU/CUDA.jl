using cuSOLVER
using LinearAlgebra

m = 15
n = 10

@testset "gesvd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, n)
    d_A = CuMatrix(A)
    U, Σ, Vt = cuSOLVER.gesvd!('A', 'A', d_A)
    @test A ≈ collect(U[:,1:n] * Diagonal(Σ) * Vt)
    @test LAPACK.gesvd!('N', 'N', CuMatrix(A))[2] ≈ Σ

    A_nan = copy(A)
    A_nan[3, 4] = NaN
    @test_throws LAPACKException cuSOLVER.gesvd!('S', 'S', CuMatrix(A_nan))

    for jobu in ('A', 'S', 'N', 'O')
        for jobvt in ('A', 'S', 'N', 'O')
            (jobu == 'A') && (jobvt == 'A') && continue
            (jobu == 'O') && (jobvt == 'O') && continue
            d_A = CuMatrix(A)
            U2, Σ2, Vt2 = cuSOLVER.gesvd!(jobu, jobvt, d_A)
            @test Σ ≈ Σ2
        end
    end
end

# `svd` is `svd!` on a copy, and is exercised below via `svd(d_A; alg)`
@testset "svd! with $alg algorithm, elty = $elty" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    alg in (cuSOLVER.QRAlgorithm(), cuSOLVER.JacobiAlgorithm()),
    (_m, _n) in ((m, n), (n, m))

    A              = rand(elty, _m, _n)
    U, S, V        = svd(A; full=true)
    d_A            = CuArray(A)

    d_U, d_S, d_V  = svd!(copy(d_A); full=true, alg=alg)
    h_S            = collect(d_S)
    h_U            = collect(d_U)
    h_V            = collect(d_V)
    @test abs.(h_U'h_U) ≈ I
    @test abs.(h_U[:,1:min(_m,_n)]'U[:,1:min(_m,_n)]) ≈ I
    @test collect(svdvals(d_A; alg=alg)) ≈ svdvals(A)
    @test svdvals!(copy(d_A); alg=alg) == svdvals(d_A; alg=alg)
    @test abs.(h_V'*h_V) ≈ I
    @test abs.(h_V[:,1:min(_m,_n)]'*V[:,1:min(_m,_n)]) ≈ I
    @test collect(d_U'*d_A*d_V) ≈ U'*A*V
    @test collect(svd(d_A; alg=alg).V') == h_V[:,1:min(_m,_n)]'
end

@testset "gesvdj convergence, elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = CuArray(randn(elty, 64, 64))
    @test_throws LAPACKException cuSOLVER.gesvdj!('V', true, copy(A); max_sweeps=2)
end

@testset "gesvdj with numerically zero singular values, elty = $elty" for
    elty in [Float32, Float64, ComplexF32, ComplexF64]

    # singular values decaying to 1e-20, so most of them are zero at working precision
    n = 128
    W = elty <: Complex ? ComplexF64 : Float64
    Q(n) = Matrix(qr(randn(W, n, n)).Q)
    s = [1e-20^((i - 1) / (n - 1)) for i in 1:n]
    A = elty.(Q(n) * Diagonal(s) * Q(n)')
    tol = 2000 * eps(real(elty))

    U, S, V = svd(CuArray(A); alg=cuSOLVER.JacobiAlgorithm())
    @test norm(collect(U) * Diagonal(collect(S)) * collect(V)' - A) / norm(A) < tol
    @test collect(S) ≈ svdvals(A)
    @test collect(svdvals(CuArray(A); alg=cuSOLVER.JacobiAlgorithm())) ≈ svdvals(A)

    # the default tolerance is reached in at most about 25 sweeps, a tolerance of eps
    # often not in 100
    U, S, V = cuSOLVER.gesvdj!('V', true, CuArray(A); max_sweeps=40)
    @test norm(collect(U) * Diagonal(collect(S)) * collect(V)' - A) / norm(A) < tol
    S = cuSOLVER.gesvdj!('N', true, CuArray(A); max_sweeps=40)[2]
    @test collect(S) ≈ svdvals(A)
end

@testset "gesvdj workspace limit, elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # the query only looks at the dimensions, so a small buffer can stand in for A
    buf = CuVector{elty}(undef, 1)
    fake(m, n) = unsafe_wrap(CuArray, pointer(buf), (m, n))
    @test cuSOLVER.gesvdj_fits('V', true, fake(64, 64))
    @test cuSOLVER.gesvdj_fits('N', true, fake(64, 64))
    # the query fails outright
    @test !cuSOLVER.gesvdj_fits('V', true, fake(65536, 65536))
    # the query succeeds, but reports a workspace size of 0
    @test !cuSOLVER.gesvdj_fits('V', true, fake(40000, 40000))
    @test !cuSOLVER.gesvdj_fits('N', true, fake(40000, 40000))
    @test_throws ArgumentError cuSOLVER.gesvdj!('V', true, fake(40000, 40000))
end

@testset "svd with QR algorithm, elty = $elty" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    full in (false, true),
    (_m, _n) in ((m, n), (n, m), (n, n))

    A = rand(elty, _m, _n)
    k = min(_m, _n)
    F = svd(CuArray(A); full, alg=cuSOLVER.QRAlgorithm())
    @test size(F.U) == (_m, full ? _m : k)
    @test size(F.Vt) == (full ? _n : k, _n)
    @test collect(F.S) ≈ svdvals(A)
    @test collect(F.U' * F.U) ≈ I
    @test collect(F.Vt * F.Vt') ≈ I
    @test collect(F.U[:, 1:k] * Diagonal(F.S) * F.Vt[1:k, :]) ≈ A
    @test collect(svdvals(CuArray(A); alg=cuSOLVER.QRAlgorithm())) ≈ svdvals(A)
end

# Check that constant propagation works
let
    _svd(A) = svd(A; alg=cuSOLVER.QRAlgorithm())
    @inferred _svd(CuArray(rand(Float32, 4, 4)))
end

@testset "batched svd! with $alg algorithm, elty = $elty" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    alg in (cuSOLVER.JacobiAlgorithm(), cuSOLVER.ApproximateAlgorithm()),
    (_m, _n, _b) in ((m, n, n), (n, m, n), (33, 33, 1))

    A              = rand(elty, _m, _n, _b)
    d_A            = CuArray(A)
    r = min(_m, _n)

    if (_m >= _n && alg == cuSOLVER.ApproximateAlgorithm()) || (_m <= 32 && _n <= 32 && alg == cuSOLVER.JacobiAlgorithm())
        d_U, d_S, d_V  = svd!(copy(d_A); full=true, alg=alg)
        h_S            = collect(d_S)
        h_U            = collect(d_U)
        h_V            = collect(d_V)
        for i=1:_b
            U, S, V = svd(A[:,:,i]; full=true)
            @test abs.(h_U[:,:,i]'*h_U[:,:,i]) ≈ I
            @test abs.(h_U[:,1:min(_m,_n),i]'U[:,1:min(_m,_n)]) ≈ I
            @test collect(svdvals(d_A; alg=alg))[:,i] ≈ svdvals(A[:,:,i])
            @test abs.(h_V[:,:,i]'*h_V[:,:,i]) ≈ I
            @test collect(d_U[:,:,i]'*d_A[:,:,i]*d_V[:,:,i])[1:r,1:r] ≈ (U'*A[:,:,i]*V)[1:r,1:r]
        end
    else
        @test_throws ArgumentError svd(d_A; alg=alg)
    end
end

@testset "gesvdaStridedBatched failure, elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, 8, 8, 3)
    A[:, end, 2] .= 0
    # older versions return NaNs without reporting the failure
    if cuSOLVER.version() >= v"11.7.3"
        @test_throws LAPACKException svd(CuArray(A); alg=cuSOLVER.ApproximateAlgorithm())
    end
    # without singular vectors, and when asking for fewer singular values than the rank,
    # the decomposition succeeds
    S = svdvals(CuArray(A); alg=cuSOLVER.ApproximateAlgorithm())
    @test collect(S[:, 2]) ≈ svdvals(A[:, :, 2])
    U, S, V = cuSOLVER.gesvdaStridedBatched!('V', CuArray(A); rank=4)
    @test collect(S[:, 2]) ≈ svdvals(A[:, :, 2])[1:4]
end

@testset "svd of $wrap($elty), uplo = $uplo" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    wrap in (Hermitian, Symmetric), uplo in (:U, :L)

    elty <: Complex && wrap === Symmetric && continue
    # a singular matrix with eigenvalues of both signs
    A = rand(elty, m, m)
    A = A + A' - 2I
    A[:, end] .= 0
    A[end, :] .= 0
    h_A = wrap(A, uplo)
    d_A = wrap(CuArray(A), uplo)

    F = svd(d_A)
    h_U, h_S, h_Vt = collect(F.U), collect(F.S), collect(F.Vt)
    @test h_S ≈ svdvals(h_A)
    @test issorted(h_S; rev=true)
    @test h_U * Diagonal(h_S) * h_Vt ≈ h_A
    @test h_U' * h_U ≈ I
    @test h_Vt * h_Vt' ≈ I
    @test collect(svdvals(d_A)) ≈ svdvals(h_A)
end

@testset "2-opnorm($sz x $elty)" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    sz in [(2, 0), (2, 3)]

    A = rand(elty, sz)
    d_A = CuArray(A)
    @test opnorm(A, 2) ≈ opnorm(d_A, 2)
end

@testset "Promotion from elty = $elty" for elty in [Float16, ComplexF16, Int32, Int64, Complex{Int32}, Complex{Int64}]
    @testset "svd with $alg algorithm" for
        alg in (cuSOLVER.QRAlgorithm(), cuSOLVER.JacobiAlgorithm()),
        (_m, _n) in ((m, n), (n, m))

        d_A = CuArray(rand(elty, _m, _n))
        d_Af = promote_type(Float32, elty).(d_A)

        @test svd(d_A; alg=alg) == svd(d_Af; alg=alg)
        @test svdvals(d_A; alg=alg) == svdvals(d_Af; alg=alg)
    end
end
