using NNlib

@testset "batched_mul" begin
    using NNlib: batched_mul, batched_mul!, batched_vec, batched_adjoint, batched_transpose

    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = batched_mul(A, B)
    @test CuArray(C) ≈ batched_mul(CuArray(A), CuArray(B))

    Ct = batched_mul(batched_transpose(A), B)
    @test CuArray(Ct) ≈ batched_mul(batched_transpose(CuArray(A)), CuArray(B))

    Ca = batched_mul(A, batched_adjoint(B))
    @test CuArray(Ca) ≈ batched_mul(CuArray(A), batched_adjoint(CuArray(B)))

    # 5-arg batched_mul!
    C .= pi
    batched_mul!(C, A, B, 2f0, 3f0)
    cuCpi = CuArray(similar(C)) .= pi
    @test CuArray(C) ≈ batched_mul!(cuCpi, CuArray(A), CuArray(B), 2f0, 3f0)

    # PermutedDimsArray
    @test CuArray(Ct) ≈ batched_mul(PermutedDimsArray(CuArray(A), (2,1,3)), CuArray(B))

    D = permutedims(B, (1,3,2))
    Cp = batched_mul(batched_adjoint(A), B)
    @test CuArray(Cp) ≈ batched_mul(batched_adjoint(CuArray(A)), PermutedDimsArray(CuArray(D), (1,3,2)))

    # Methods which reshape
    M = randn(Float32, 3,3)

    Cm = batched_mul(A, M)
    @test CuArray(Cm) ≈ batched_mul(CuArray(A), CuArray(M))

    Cv = batched_vec(permutedims(A,(3,1,2)), M)
    @test CuArray(Cv) ≈ batched_vec(PermutedDimsArray(CuArray(A),(3,1,2)), CuArray(M))
end

@testset "NNlib storage_type etc." begin
    using LinearAlgebra
    using NNlib: is_strided, are_strided, storage_type

    M = cu(ones(10,10))

    @test is_strided(M)
    @test is_strided(view(M, 1:2:5,:))
    @test is_strided(PermutedDimsArray(M, (2,1)))

    @test !is_strided(reshape(view(M, 1:2:10,:), 10,:))
    @test !is_strided((M .+ im)')
    @test !is_strided(Diagonal(cu(ones(3))))

    @test storage_type(M) == CuArray{Float32,2}
    @test storage_type(reshape(view(M, 1:2:10,:), 10,:)) == CuArray{Float32,2}

end

@testset "Broadcast Fix" begin
  if CUDA.has_cudnn()
    @test testf(x -> logσ.(x), rand(5))
  end
end
