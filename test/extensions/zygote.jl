using GPUArraysCore: GPUArraysCore
using CUDA
using Zygote

function call_rand(v::AbstractVector{T}) where {T}
    randn(T, 4,4) * v[1:4]
end
function call_rand(v::GPUArraysCore.AbstractGPUVector{T}) where {T}
    CUDA.randn(T, 4,4) * v[1:4]
end

@testset "randn" begin
    v_orig = collect(1.0f0:10.0f0)
    mb = call_rand(v_orig)
    v = CuArray(v_orig)
    m = call_rand(v)
    gr = Zygote.gradient(v -> sum(call_rand(v)), v) 
    @test gr[1:4] .!= 0
end
