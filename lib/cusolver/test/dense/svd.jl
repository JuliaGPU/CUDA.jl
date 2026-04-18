using cuSOLVER
using LinearAlgebra

m = 15
n = 10

@testset "gesvd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, n)
    d_A = CuMatrix(A)
    U, Σ, Vt = cuSOLVER.gesvd!('A', 'A', d_A)
    @test A ≈ collect(U[:,1:n] * Diagonal(Σ) * Vt)

    for jobu in ('A', 'S', 'N', 'O')
        for jobvt in ('A', 'S', 'N', 'O')
            (jobu == 'A') && (jobvt == 'A') && continue
            (jobu == 'O') && (jobvt == 'O') && continue
            d_A = CuMatrix(A)
            U2, Σ2, Vt2 = cuSOLVER.gesvd!(jobu, jobvt, d_A)
            @test Σ ≈ Σ2
            d_A = CuMatrix(A)
            U2, Σ2, Vt2 = LAPACK.gesvd!(jobu, jobvt, d_A)
            @test Σ ≈ Σ2
        end
    end
end

@testset "$svd_f with $alg algorithm, elty = $elty" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    svd_f in (svd, svd!),
    alg in (cuSOLVER.QRAlgorithm(), cuSOLVER.JacobiAlgorithm()),
    (_m, _n) in ((m, n), (n, m))

    A              = rand(elty, _m, _n)
    U, S, V        = svd(A; full=true)
    d_A            = CuArray(A)

    if _m > _n || alg == cuSOLVER.JacobiAlgorithm()
        d_U, d_S, d_V  = svd_f(copy(d_A); full=true, alg=alg)
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
    else
        @test_throws ArgumentError svd(d_A; alg=alg)
    end
end

# Check that constant propagation works
let
    _svd(A) = svd(A; alg=cuSOLVER.QRAlgorithm())
    @inferred _svd(CuArray(rand(Float32, 4, 4)))
end

@testset "batched $svd_f with $alg algorithm, elty = $elty" for
    elty in [Float32, Float64, ComplexF32, ComplexF64],
    svd_f in (svd, svd!),
    alg in (cuSOLVER.JacobiAlgorithm(), cuSOLVER.ApproximateAlgorithm()),
    (_m, _n, _b) in ((m, n, n), (n, m, n), (33, 33, 1))

    A              = rand(elty, _m, _n, _b)
    d_A            = CuArray(A)
    r = min(_m, _n)

    if (_m >= _n && alg == cuSOLVER.ApproximateAlgorithm()) || (_m <= 32 && _n <= 32 && alg == cuSOLVER.JacobiAlgorithm())
        d_U, d_S, d_V  = svd_f(copy(d_A); full=true, alg=alg)
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

        if _m > _n || alg == cuSOLVER.JacobiAlgorithm()
            @test svd(d_A; alg=alg) == svd(d_Af; alg=alg)
            @test svdvals(d_A; alg=alg) == svdvals(d_Af; alg=alg)
        else
            @test_throws ArgumentError svd(d_A; alg=alg)
            @test_throws ArgumentError svdvals(d_A; alg=alg)
        end
    end
end
