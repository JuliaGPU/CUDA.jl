using cuSOLVER
using LinearAlgebra

m = 15
n = 10

# Adapted from LinearAlgebra.sorteig!().
# Warning: not very efficient, but works.
eigsortby(λ::Real) = λ
eigsortby(λ::Complex) = (real(λ), imag(λ))
function sorteig!(λ::AbstractVector, X::AbstractMatrix, sortby::Union{Function, Nothing} = eigsortby)
    if sortby !== nothing
        p = sortperm(λ; by = sortby)
        λ .= λ[p]
        X .= X[:, p]
    end
    return λ, X
end
sorteig!(λ::AbstractVector, sortby::Union{Function, Nothing} = eigsortby) = sortby === nothing ? λ : sort!(λ, by = sortby)

# Note: Xgeev was introduced in CUDA 12.6.2 / CUSOLVER 11.7.1
if cuSOLVER.version() >= v"11.7.1"
    @testset "geev! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        local d_W, d_V

        A              = rand(elty, m, m)
        d_A            = CuArray(A)
        Eig            = eigen(A)
        d_eig          = eigen(d_A)
        sorteig!(d_eig.values, d_eig.vectors)
        @test Eig.values ≈ collect(d_eig.values)
        h_V            = collect(d_eig.vectors)
        h_V⁻¹          = inv(h_V)
        @test abs.(h_V⁻¹*Eig.vectors) ≈ I

        A              = rand(elty, m, m)
        d_A            = CuArray(A)
        W              = eigvals(A)
        d_W            = eigvals(d_A)
        sorteig!(d_W)
        @test W        ≈ collect(d_W)

        A              = rand(elty, m, m)
        d_A            = CuArray(A)
        V              = eigvecs(A)
        d_W            = eigvals(d_A)
        d_V            = eigvecs(d_A)
        sorteig!(d_W, d_V)
        V⁻¹            = inv(V)
        @test abs.(V⁻¹*collect(d_V)) ≈ I
    end
end

@testset "syevd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A              = rand(elty, m, m)
    A             += A'
    d_A            = CuArray(A)
    local d_W, d_V
    if elty <: Complex
        d_W, d_V   = cuSOLVER.heevd!('V', 'U', d_A)
        d_W_b, d_V_b  = LAPACK.syev!('V', 'U', CuArray(A))
        @test d_W ≈ d_W_b
        @test d_V ≈ d_V_b
        d_W_c, d_V_c  = LAPACK.syevd!('V', 'U', CuArray(A))
        @test d_W ≈ d_W_c
        @test d_V ≈ d_V_c
    else
        d_W, d_V   = cuSOLVER.syevd!('V', 'U', d_A)
        d_W_b, d_V_b  = LAPACK.syev!('V', 'U', CuArray(A))
        @test d_W ≈ d_W_b
        @test d_V ≈ d_V_b
        d_W_c, d_V_c  = LAPACK.syevd!('V', 'U', CuArray(A))
        @test d_W ≈ d_W_c
        @test d_V ≈ d_V_c
    end
    h_W            = collect(d_W)
    h_V            = collect(d_V)
    Eig            = eigen(A)
    @test Eig.values ≈ h_W
    @test abs.(Eig.vectors'*h_V) ≈ I
    d_A            = CuArray(A)
    if elty <: Complex
        d_W   = cuSOLVER.heevd!('N', 'U', d_A)
    else
        d_W   = cuSOLVER.syevd!('N', 'U', d_A)
    end
    h_W            = collect(d_W)
    @test Eig.values ≈ h_W

    A              = rand(elty, m, m)
    A             += A'
    d_A            = CuArray(A)
    Eig            = eigen(LinearAlgebra.Hermitian(A))
    d_eig          = eigen(d_A)
    sorteig!(d_eig.values, d_eig.vectors)
    @test Eig.values ≈ collect(d_eig.values)
    d_eig          = eigen(LinearAlgebra.Hermitian(d_A))
    @test Eig.values ≈ collect(d_eig.values)
    h_V            = collect(d_eig.vectors)
    @test abs.(Eig.vectors'*h_V) ≈ I
    if elty <: Real
        Eig            = eigen(LinearAlgebra.Symmetric(A))
        d_eig          = eigen(LinearAlgebra.Symmetric(d_A))
        @test Eig.values ≈ collect(d_eig.values)
        h_V            = collect(d_eig.vectors)
        @test abs.(Eig.vectors'*h_V) ≈ I
    end

    A              = rand(elty, m, m)
    A             += A'
    d_A            = CuArray(A)
    W              = eigvals(LinearAlgebra.Hermitian(A))
    d_W            = eigvals(d_A)
    sorteig!(d_W)
    @test W        ≈ collect(d_W)
    d_W            = eigvals(LinearAlgebra.Hermitian(d_A))
    @test W        ≈ collect(d_W)
    if elty <: Real
        W              = eigvals(LinearAlgebra.Symmetric(A))
        d_W            = eigvals(LinearAlgebra.Symmetric(d_A))
        @test W        ≈ collect(d_W)
    end

    A              = rand(elty, m, m)
    A             += A'
    d_A            = CuArray(A)
    V              = eigvecs(LinearAlgebra.Hermitian(A))
    d_W            = eigvals(d_A)
    d_V            = eigvecs(d_A)
    sorteig!(d_W, d_V)
    h_V            = collect(d_V)
    @test abs.(V'*h_V) ≈ I
    d_V            = eigvecs(LinearAlgebra.Hermitian(d_A))
    h_V            = collect(d_V)
    @test abs.(V'*h_V) ≈ I
    if elty <: Real
        V              = eigvecs(LinearAlgebra.Symmetric(A))
        d_V            = eigvecs(LinearAlgebra.Symmetric(d_A))
        h_V            = collect(d_V)
        @test abs.(V'*h_V) ≈ I
    end
end

@testset "sygvd! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A              = rand(elty, m, m)
    B              = rand(elty, m, m)
    A              = A*A'+I # posdef
    B              = B*B'+I # posdef
    d_A            = CuArray(A)
    d_B            = CuArray(B)
    local d_W, d_VA, d_VB
    if elty <: Complex
        d_W, d_VA, d_VB = cuSOLVER.hegvd!(1, 'V', 'U', d_A, d_B)
        d_W2, d_VA2, d_VB2 = LAPACK.sygvd!(1, 'V', 'U', CuArray(A), CuArray(B))
        @test d_W2 ≈ d_W
        @test d_VA2 ≈ d_VA
        @test d_VB2 ≈ d_VB
    else
        d_W, d_VA, d_VB = cuSOLVER.sygvd!(1, 'V', 'U', d_A, d_B)
        d_W2, d_VA2, d_VB2 = LAPACK.sygvd!(1, 'V', 'U', CuArray(A), CuArray(B))
        @test d_W2 ≈ d_W
        @test d_VA2 ≈ d_VA
        @test d_VB2 ≈ d_VB
    end
    h_W            = collect(d_W)
    h_VA           = collect(d_VA)
    h_VB           = collect(d_VB)
    Eig            = eigen(Hermitian(A), Hermitian(B))
    @test Eig.values ≈ h_W
    @test A*h_VA ≈ B*h_VA*Diagonal(h_W) rtol=1e-4
    # test normalization condition for eigtype 1
    @test abs.(h_VA'B*h_VA) ≈ Matrix(one(elty)*I, m, m)
    d_A            = CuArray(A)
    d_B            = CuArray(B)
    if elty <: Complex
        d_W   = cuSOLVER.hegvd!(1, 'N', 'U', d_A, d_B)
    else
        d_W   = cuSOLVER.sygvd!(1, 'N', 'U', d_A, d_B)
    end
    h_W            = collect(d_W)
    @test Eig.values ≈ h_W
    d_B            = CuArray(rand(elty, m+1, m+1))
    if elty <: Complex
        @test_throws DimensionMismatch cuSOLVER.hegvd!(1, 'N', 'U', d_A, d_B)
    else
        @test_throws DimensionMismatch cuSOLVER.sygvd!(1, 'N', 'U', d_A, d_B)
    end
end

@testset "syevj! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A              = rand(elty, m, m)
    B              = rand(elty, m, m)
    A              = A*A'+I # posdef
    B              = B*B'+I # posdef
    d_A            = CuArray(A)
    d_B            = CuArray(B)
    local d_W, d_VA, d_VB
    if elty <: Complex
        d_W, d_VA, d_VB = cuSOLVER.hegvj!(1, 'V', 'U', d_A, d_B)
    else
        d_W, d_VA, d_VB = cuSOLVER.sygvj!(1, 'V', 'U', d_A, d_B)
    end
    h_W            = collect(d_W)
    h_VA           = collect(d_VA)
    h_VB           = collect(d_VB)
    Eig            = eigen(Hermitian(A), Hermitian(B))
    @test Eig.values ≈ h_W
    @test A*h_VA ≈ B*h_VA*Diagonal(h_W) rtol=1e-4
    # test normalization condition for eigtype 1
    @test abs.(h_VA'B*h_VA) ≈ Matrix(one(elty)*I, m, m)
    d_A            = CuArray(A)
    d_B            = CuArray(B)
    if elty <: Complex
        d_W   = cuSOLVER.hegvj!(1, 'N', 'U', d_A, d_B)
    else
        d_W   = cuSOLVER.sygvj!(1, 'N', 'U', d_A, d_B)
    end
    h_W            = collect(d_W)
    @test Eig.values ≈ h_W
end

@testset "syevjBatched! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    # Generate a random symmetric/hermitian matrix
    A = rand(elty, m, m, n)
    A += permutedims(A, (2, 1, 3))
    d_A = CuArray(A)

    local d_W, d_V
    if elty <: Complex
        d_W, d_V   = cuSOLVER.heevjBatched!('V', 'U', d_A)
    else
        d_W, d_V   = cuSOLVER.syevjBatched!('V', 'U', d_A)
    end

    h_W   = collect(d_W)
    h_V   = collect(d_V)

    for i = 1:n
        Eig = eigen(LinearAlgebra.Hermitian(A[:,:,i]))
        @test Eig.values ≈ h_W[:,i]
        @test abs.(Eig.vectors'*h_V[:,:,i]) ≈ I
    end

    # without eigenvectors
    d_A = CuArray(A)
    local d_W2
    if elty <: Complex
        d_W2   = cuSOLVER.heevjBatched!('N', 'U', d_A)
    else
        d_W2   = cuSOLVER.syevjBatched!('N', 'U', d_A)
    end

    h_W2   = collect(d_W2)

    for i = 1:n
        Eig = eigen(LinearAlgebra.Hermitian(A[:,:,i]))
        @test Eig.values ≈ h_W2[:,i]
    end
end
