using NNlib

@testset "batched_mul" begin
    using NNlib: batched_mul, batched_adjoint, batched_transpose

    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = batched_mul(A, B)
    @test CuArray(C) ≈ batched_mul(CuArray(A), CuArray(B))

    Ct = batched_mul(batched_transpose(A), B)
    @test CuArray(Ct) ≈ batched_mul(batched_transpose(CuArray(A)), CuArray(B))

    Ca = batched_mul(A, batched_adjoint(B))
    @test CuArray(Ca) ≈ batched_mul(CuArray(A), batched_adjoint(CuArray(B)))
end

@testset "Broadcast Fix" begin
  if CUDA.has_cudnn()
    @test testf(x -> logσ.(x), rand(5))
  end
end
