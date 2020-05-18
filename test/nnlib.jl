using NNlib

@testset "NNlib" begin

@testset "batched_mul" begin
    using NNlib: batched_mul, batched_adjoint, batched_transpose

    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = batched_mul(A, B)
    @test cu(C) ≈ batched_mul(cu(A), cu(B))

    Ct = batched_mul(batched_transpose(A), B)
    @test cu(Ct) ≈ batched_mul(batched_transpose(cu(A)), cu(B))

    Ca = batched_mul(A, batched_adjoint(B))
    @test cu(Ca) ≈ batched_mul(cu(A), batched_adjoint(cu(B)))
end

@testset "Broadcast Fix" begin
  if CUDA.has_cudnn()
    @test testf(x -> logσ.(x), rand(5))
  end
end

end
