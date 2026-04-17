using cuSOLVER
using LinearAlgebra

n = 10

if cuSOLVER.version() >= v"11.7.1"
    @testset "geev! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        A = rand(elty, n, n)
        d_A = CuMatrix(A)
        d_B = copy(d_A)
        W, VL, VR = cuSOLVER.Xgeev!('N', 'V', d_A)
        if elty <: Complex
            @test d_B * VR ≈ VR * Diagonal(W)
        else
            h_W = collect(W)
            i = 1
            while i <= n
                if h_W[i].im ≈ zero(elty)
                    @test d_B * VR[:,i] ≈ h_W[i].re * VR[:,i]
                    i = i + 1
                else
                    V1 = VR[:,i] + im * VR[:,i+1]
                    @test d_B * V1 ≈ h_W[i] * V1
                    V2 = VR[:,i] - im * VR[:,i+1]
                    @test d_B * V2 ≈ h_W[i+1] * V2
                    i = i + 2
                end
            end
        end
    end

    @testset "syevBatched! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        batch_size = 5
        for uplo in ('L', 'U')
            (cuSOLVER.version() < v"11.7.2") && (uplo == 'L') && (elty == ComplexF32) && continue

            A = rand(elty, n, n, batch_size)
            B = rand(elty, n, n, batch_size)
            for i in 1:batch_size
                S = rand(elty, n, n)
                S = S * S' + I
                B[:, :, i] .= S
                S = uplo == 'L' ? tril(S) : triu(S)
                A[:, :, i] .= S
            end
            d_A = CuArray(A)
            d_W, d_V = cuSOLVER.XsyevBatched!('V', uplo, d_A)
            W = collect(d_W)
            V = collect(d_V)
            for i in 1:batch_size
                Bᵢ = B[:, :, i]
                Wᵢ = Diagonal(W[:, i])
                Vᵢ = V[:, :, i]
                @test Bᵢ * Vᵢ ≈ Vᵢ * Diagonal(Wᵢ)
            end

            d_A = CuArray(A)
            d_W = cuSOLVER.XsyevBatched!('N', uplo, d_A)
        end
    end

    @testset "syevBatched! updated elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        batch_size = 5
        for uplo in ('L', 'U')
            (cuSOLVER.version() < v"11.7.2") && (uplo == 'L') && (elty == ComplexF32) && continue

            A = rand(elty, n, n * batch_size)
            B = rand(elty, n, n * batch_size)
            for i = 1:batch_size
                S = rand(elty, n, n)
                S = S * S' + I
                B[:,(i-1)*n+1:i*n] .= S
                S = uplo == 'L' ? tril(S) : triu(S)
                A[:,(i-1)*n+1:i*n] .= S
            end
            d_A = CuMatrix(A)
            d_W, d_V = cuSOLVER.XsyevBatched!('V', uplo, d_A)
            W = collect(d_W)
            V = collect(d_V)
            for i = 1:batch_size
                Bᵢ = B[:,(i-1)*n+1:i*n]
                Wᵢ = Diagonal(W[(i-1)*n+1:i*n])
                Vᵢ = V[:,(i-1)*n+1:i*n]
                @test Bᵢ * Vᵢ ≈ Vᵢ * Diagonal(Wᵢ)
            end

            d_A = CuMatrix(A)
            d_W = cuSOLVER.XsyevBatched!('N', uplo, d_A)
        end
    end
end

@testset "syevd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    for uplo in ('L', 'U')
        A = rand(elty, n, n)
        B = A + A'
        A = uplo == 'L' ? tril(B) : triu(B)
        d_A = CuMatrix(A)
        W, V = cuSOLVER.Xsyevd!('V', uplo, d_A)
        @test B ≈ collect(V * Diagonal(W) * V')

        d_A = CuMatrix(A)
        d_W = cuSOLVER.Xsyevd!('N', uplo, d_A)
    end
end

@testset "syevdx! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    R = real(elty)
    Σ = [i*one(R) for i = 1:10]
    B = rand(elty, 10, 10)
    F = qr(B)
    A = F.Q * Diagonal(Σ) * F.Q'
    for uplo in ('L', 'U')
        h_A = uplo == 'L' ? tril(A) : triu(A)
        d_A = CuMatrix{elty}(h_A)

        d_W, d_V, neig = cuSOLVER.Xsyevdx!('V', 'A', uplo, d_A, vl=3.5, vu=7.5, il=1, iu=3)
        @test neig == 10
        @test collect(d_W) ≈ Σ
        @test A ≈ collect(d_V * Diagonal(d_W) * d_V')

        d_W, neig = cuSOLVER.Xsyevdx!('N', 'I', uplo, d_A, vl=3.5, vu=7.5, il=1, iu=3)
        @test neig == 3

        d_W, neig = cuSOLVER.Xsyevdx!('N', 'V', uplo, d_A, vl=3.5, vu=7.5, il=1, iu=3)
    end
end
