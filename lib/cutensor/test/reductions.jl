@testset "reductions" begin

using cuTENSOR: reduce!

using LinearAlgebra, Random

eltypes = [(Float16, Float16),
           (Float32, Float32),
           (Float64, Float64),
           (ComplexF32, ComplexF32),
           (ComplexF64, ComplexF64)]

@testset for NA=2:5, NC = 1:NA-1
    @testset for (eltyA, eltyC) in eltypes
        # setup
        eltyD = eltyC
        dmax = 2^div(18,NA)
        dims = rand(2:dmax, NA)
        p = randperm(NA)
        indsA = collect(('a':'z')[1:NA])
        indsC = indsA[p][1:NC]
        dimsA = dims
        dimsC = dims[p][1:NC]
        A = rand(eltyA, (dimsA...,))
        dA = CuArray(A)
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)

        opA = cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opReduce = cuTENSOR.OP_ADD
        # simple case
        dC = reduce!(1, dA, indsA, opA, 0, dC, indsC, opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))

        # using integers as indices
        dC = reduce!(1, dA, collect(1:NA), opA, 0, dC, p[1:NC], opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))

        # multiplication as reduction operator
        opReduce = cuTENSOR.OP_MUL
        dC = reduce!(1, dA, indsA, opA, 0, dC, indsC, opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            prod(permutedims(A, p); dims = ((NC+1:NA)...,)) atol=eps(Float16) rtol=Base.rtoldefault(Float16)

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? cuTENSOR.OP_CONJ : cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opReduce = cuTENSOR.OP_ADD
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)
        α = rand(eltyC)
        γ = rand(eltyC)
        dC = reduce!(α, dA, indsA, opA, γ, dC, indsC, opC, opReduce)
        @test reshape(collect(dC), (dimsC..., ones(Int,NA-NC)...)) ≈
            α .* conj.(sum(permutedims(A, p); dims = ((NC+1:NA)...,))) .+ γ .* C
    end
end

end
