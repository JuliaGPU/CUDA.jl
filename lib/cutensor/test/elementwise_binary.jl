@testset "elementwise binary" begin

using cuTENSOR: elementwise_binary_execute!

using LinearAlgebra

eltypes = [(Float16, Float16),
           (Float32, Float32),
           (Float64, Float64),
           (ComplexF32, ComplexF32),
           (ComplexF64, ComplexF64),
           (ComplexF64, ComplexF32),
           (Float32, Float16),
           (Float64, Float32)]

@testset for N=2:5
    @testset for (eltyA, eltyC) in eltypes
        # setup
        eltyD = eltyC
        dmax  = 2^div(18,N)
        dims  = rand(2:dmax, N)
        p     = randperm(N)
        indsA = collect(('a':'z')[1:N])
        indsC = indsA[p]
        dimsA = dims
        dimsC = dims[p]
        A     = rand(eltyA, dimsA...)
        dA    = CuArray(A)
        C     = rand(eltyC, dimsC...)
        dC    = CuArray(C)

        # simple case
        opA   = cuTENSOR.OP_IDENTITY
        opC   = cuTENSOR.OP_IDENTITY
        dD    = similar(dC, eltyD)
        opAC  = cuTENSOR.OP_ADD
        dD    = elementwise_binary_execute!(1, dA, indsA, opA, 1, dC, indsC, opC, dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C

        # using integers as indices
        dD = elementwise_binary_execute!(1, dA, 1:N, opA, 1, dC, p, opC, dD, p, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C

        # multiplication as binary operator
        opAC = cuTENSOR.OP_MUL
        dD = elementwise_binary_execute!(1, dA, indsA, opA, 1, dC, indsC, opC, dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .* C

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? cuTENSOR.OP_CONJ : cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opAC = cuTENSOR.OP_ADD
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = elementwise_binary_execute!(α, dA, indsA, opA, γ, dC, indsC, opC, dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, p)) .+ γ .* C

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? cuTENSOR.OP_IDENTITY : cuTENSOR.OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? cuTENSOR.OP_CONJ : cuTENSOR.OP_IDENTITY
        opAC = eltyD <: Complex ? cuTENSOR.OP_ADD : cuTENSOR.OP_MAX
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = elementwise_binary_execute!(α, dA, indsA, opA, γ, dC, indsC, opC, dC, indsC, opAC)
        D = collect(dC)
        if eltyD <: Complex
            if eltyA <: Complex
                @test D ≈ α .* permutedims(A, p) .+ γ .* conj.(C)
            else
                @test D ≈ α .* sqrt.(eltyD.(permutedims(A, p))) .+ γ .* conj.(C)
            end
        else
            @test D ≈ max.(α .* sqrt.(eltyD.(permutedims(A, p))), γ .* C)
        end

        # using CuTensor type
        dA = CuArray(A)
        dC = CuArray(C)
        ctA = CuTensor(dA, indsA)
        ctC = CuTensor(dC, indsC)
        ctD = ctA + ctC
        hD = collect(ctD.data)
        @test hD ≈ permutedims(A, p) .+ C
        ctD = ctA - ctC
        hD = collect(ctD.data)
        @test hD ≈ permutedims(A, p) .- C

        α = rand(eltyD)
        ctC_copy = copy(ctC)
        ctD = LinearAlgebra.axpy!(α, ctA, ctC_copy)
        @test ctD == ctC_copy
        hD = collect(ctD.data)
        @test hD ≈ α.*permutedims(A, p) .+ C

        γ = rand(eltyD)
        ctC_copy = copy(ctC)
        ctD = LinearAlgebra.axpby!(α, ctA, γ, ctC_copy)
        @test ctD == ctC_copy
        hD = collect(ctD.data)
        @test hD ≈ α.*permutedims(A, p) .+ γ.*C
    end
end

end
