using LinearAlgebra

using NNlib

@testcase "NNlib" begin

@testcase "batched_mul" begin

    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = NNlib.batched_mul(A, B)
    @test CuArray(C) ≈ NNlib.batched_mul(CuArray(A), CuArray(B))

    Ct = NNlib.batched_mul(NNlib.batched_transpose(A), B)
    @test CuArray(Ct) ≈ NNlib.batched_mul(NNlib.batched_transpose(CuArray(A)), CuArray(B))

    Ca = NNlib.batched_mul(A, NNlib.batched_adjoint(B))
    @test CuArray(Ca) ≈ NNlib.batched_mul(CuArray(A), NNlib.batched_adjoint(CuArray(B)))

    # 5-arg NNlib.batched_mul!
    C .= pi
    NNlib.batched_mul!(C, A, B, 2f0, 3f0)
    cuCpi = CuArray(similar(C)) .= pi
    @test CuArray(C) ≈ NNlib.batched_mul!(cuCpi, CuArray(A), CuArray(B), 2f0, 3f0)

    # PermutedDimsArray
    @test CuArray(Ct) ≈ NNlib.batched_mul(PermutedDimsArray(CuArray(A), (2,1,3)), CuArray(B))

    D = permutedims(B, (1,3,2))
    Cp = NNlib.batched_mul(NNlib.batched_adjoint(A), B)
    @test CuArray(Cp) ≈ NNlib.batched_mul(NNlib.batched_adjoint(CuArray(A)), PermutedDimsArray(CuArray(D), (1,3,2)))

    # Methods which reshape
    M = randn(Float32, 3,3)

    Cm = NNlib.batched_mul(A, M)
    @test CuArray(Cm) ≈ NNlib.batched_mul(CuArray(A), CuArray(M))

    Cv = NNlib.batched_vec(permutedims(A,(3,1,2)), M)
    @test CuArray(Cv) ≈ NNlib.batched_vec(PermutedDimsArray(CuArray(A),(3,1,2)), CuArray(M))
end

@testcase "storage_type" begin

    M = cu(ones(10,10))

    @test NNlib.is_strided(M)
    @test NNlib.is_strided(view(M, 1:2:5,:))
    @test NNlib.is_strided(PermutedDimsArray(M, (2,1)))

    @test !NNlib.is_strided(reshape(view(M, 1:2:10,:), 10,:))
    @test !NNlib.is_strided((M .+ im)')
    @test !NNlib.is_strided(Diagonal(cu(ones(3))))

    @test NNlib.storage_type(M) == CuArray{Float32,2}
    @test NNlib.storage_type(reshape(view(M, 1:2:10,:), 10,:)) == CuArray{Float32,2}

end

@testcase "broadcast" begin
  if CUDA.has_cudnn()
    @test testf(x -> logσ.(x), rand(5))
  end
end

end
