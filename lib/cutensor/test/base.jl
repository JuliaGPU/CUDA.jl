@testset "base" begin

using LinearAlgebra, Random

@test has_cutensor()
@test cuTENSOR.version() isa VersionNumber

@testset "type basics" begin
    N = 2
    dmax = 2^div(18,N)
    dims = rand(2:dmax, N)
    p = randperm(N)
    indsA = collect(('a':'z')[1:N])
    dimsA = dims
    A = rand(Float64, dimsA...)
    dA = CuArray(A)
    p = randperm(N)
    indsA = collect(('a':'z')[1:N])
    ctA = CuTensor(dA, indsA)
    @test length(ctA) == length(A)
    @test size(ctA) == size(A)
    @test size(ctA, 1) == size(A, 1)
    @test ndims(ctA) == ndims(A)
    @test strides(ctA) == strides(A)
    @test eltype(ctA) == eltype(A)
end

end
