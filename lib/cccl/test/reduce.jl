for T in (Int32, Int64, Float32, Float64)
    x = rand(T(1):T(100), 10_000)
    @test CCCL.reduce(+, CuArray(x); init=zero(T)) == sum(x)
    @test CCCL.reduce(max, CuArray(x); init=typemin(T)) == maximum(x)
end

# arbitrary Julia functions
x = rand(Float32, 10_000)
@test CCCL.reduce((a, b) -> max(abs(a), abs(b)), CuArray(x .- 0.5f0); init=0f0) ≈
      maximum(abs, x .- 0.5f0)

# custom element types
x = [ComplexF32(rand(), rand()) for _ in 1:1000]
@test CCCL.reduce(+, CuArray(x); init=zero(ComplexF32)) ≈ sum(x)
