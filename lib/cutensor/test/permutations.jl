using cuTENSOR: permute!

using LinearAlgebra, Random

@testset "permutations" begin

eltypes = [(Float16, Float16),
           (Float16, Float32),
           (Float32, Float16),
           (Float32, Float32),
           (Float64, Float64),
           (Float32, Float64),
           (Float64, Float32),
           (ComplexF32, ComplexF32),
           (ComplexF64, ComplexF64),
           (ComplexF32, ComplexF64),
           (ComplexF64, ComplexF32),
           ]

@testset for N=2:5
    @testset for (eltyA, eltyC) in eltypes
        # setup
        dmax = 2^div(18,N)
        dims = rand(2:dmax, N)
        p = randperm(N)
        indsA = collect(('a':'z')[1:N])
        indsC = indsA[p]
        dimsA = dims
        dimsC = dims[p]
        A = rand(eltyA, dimsA...)
        dA = CuArray(A)
        dC = similar(dA, eltyC, dimsC...)

        # simple case
        opA = cuTENSOR.OP_IDENTITY
        dC = permute!(one(eltyA), dA, indsA, opA, dC, indsC)
        C  = collect(dC)
        @test C ≈ eltyC.(permutedims(A, p))

        # with scalar
        α  = rand(eltyA)
        dC = permute!(α, dA, indsA, opA, dC, indsC)
        C  = collect(dC)
        @test C ≈ α * permutedims(A, p) # approximate, floating point rounding
    end
end

end

@testset "releasing plans" begin
    a = CuArray(rand(Float32, 16, 16))
    b = similar(a)
    plan = cuTENSOR.plan_permutation(a, ['i', 'j'], cuTENSOR.OP_IDENTITY, b, ['j', 'i'])
    permute!(plan, 1, a, b)
    @test Array(b) == permutedims(Array(a))
    workspace = plan.workspace.data
    # destroying a plan may wait for the GPU, so that's deferred until reclaiming memory
    finalize(plan)
    CUDACore.pool_status(devnull)
    @test plan.handle != C_NULL
    CUDACore.reclaim()
    @test plan.handle == C_NULL
    @test workspace.freed
end
