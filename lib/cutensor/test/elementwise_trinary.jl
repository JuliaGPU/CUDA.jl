using CUDA, cuTENSOR
using LinearAlgebra

eltypes = ((Float16, Float16, Float16),
            #(Float16, Float32, Float32),
            # (Float32, Float16, Float32),
            (Float32, Float32, Float32),
            # (Float32, Float32, Float64),
            # (Float32, Float64, Float64),
            # (Float64, Float32, Float64),
            (Float64, Float64, Float64),
            (ComplexF32, ComplexF32, ComplexF32),
            (ComplexF64, ComplexF64, ComplexF64))
@testset for N=2:5
    @testset for (eltyA, eltyB, eltyC) in eltypes
        # setup
        eltyD = eltyC
        dmax = 2^div(18,N)
        dims = rand(2:dmax, N)
        pA = randperm(N)
        ipA = invperm(pA)
        pB = randperm(N)
        ipB = invperm(pB)
        indsC = collect(('a':'z')[1:N])
        dimsC = dims
        indsA = indsC[ipA]
        dimsA = dims[ipA]
        indsB = indsC[ipB]
        dimsB = dims[ipB]
        A = rand(eltyA, dimsA...)
        dA = CuArray(A)
        B = rand(eltyB, dimsB...)
        dB = CuArray(B)
        C = rand(eltyC, dimsC...)
        dC = CuArray(C)
        dD = similar(dC)

        # simple case
        opA = cuTENSOR.CUTENSOR_OP_IDENTITY
        opB = cuTENSOR.CUTENSOR_OP_IDENTITY
        opC = cuTENSOR.CUTENSOR_OP_IDENTITY
        opAB = cuTENSOR.CUTENSOR_OP_ADD
        opABC = cuTENSOR.CUTENSOR_OP_ADD
        dD = cuTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C

        # using integers as indices
        dD = cuTENSOR.elementwiseTrinary!(1, dA, ipA, opA, 1, dB, ipB, opB,
                                            1, dC, 1:N, opC, dD, 1:N, opAB, opABC)
        D = collect(dD)
        @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C

        # multiplication as binary operator
        opAB = cuTENSOR.CUTENSOR_OP_MUL
        opABC = cuTENSOR.CUTENSOR_OP_ADD
        dD = cuTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ (convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB))) .+ C

        opAB = cuTENSOR.CUTENSOR_OP_ADD
        opABC = cuTENSOR.CUTENSOR_OP_MUL
        dD = cuTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ (convert.(eltyD, permutedims(A, pA)) .+ convert.(eltyD, permutedims(B, pB))) .* C

        opAB = cuTENSOR.CUTENSOR_OP_MUL
        opABC = cuTENSOR.CUTENSOR_OP_MUL
        dD = cuTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ convert.(eltyD, permutedims(A, pA)) .*
                    convert.(eltyD, permutedims(B, pB)) .* C

        # with non-trivial coefficients and conjugation
        α = rand(eltyD)
        β = rand(eltyD)
        γ = rand(eltyD)
        opA = eltyA <: Complex ? cuTENSOR.CUTENSOR_OP_CONJ :
                                cuTENSOR.CUTENSOR_OP_IDENTITY
        opAB = cuTENSOR.CUTENSOR_OP_ADD
        opABC = cuTENSOR.CUTENSOR_OP_ADD
        dD = cuTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, pA)) .+ β .* permutedims(B, pB) .+ γ .* C

        opB = eltyB <: Complex ? cuTENSOR.CUTENSOR_OP_CONJ :
                                cuTENSOR.CUTENSOR_OP_IDENTITY
        opAB = cuTENSOR.CUTENSOR_OP_ADD
        opABC = cuTENSOR.CUTENSOR_OP_ADD
        dD = cuTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, pA)) .+
                    β .* conj.(permutedims(B, pB)) .+ γ .* C

        opA = cuTENSOR.CUTENSOR_OP_IDENTITY
        opAB = cuTENSOR.CUTENSOR_OP_MUL
        opABC = cuTENSOR.CUTENSOR_OP_ADD
        dD = cuTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* permutedims(A, pA) .* β .* conj.(permutedims(B, pB)) .+ γ .* C

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? cuTENSOR.CUTENSOR_OP_IDENTITY :
                                cuTENSOR.CUTENSOR_OP_SQRT
        opB = eltyB <: Complex ? cuTENSOR.CUTENSOR_OP_IDENTITY :
                                cuTENSOR.CUTENSOR_OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? cuTENSOR.CUTENSOR_OP_CONJ :
                                cuTENSOR.CUTENSOR_OP_IDENTITY
        opAB = eltyD <: Complex ? cuTENSOR.CUTENSOR_OP_MUL :
                                cuTENSOR.CUTENSOR_OP_MIN
        opABC = eltyD <: Complex ? cuTENSOR.CUTENSOR_OP_ADD :
                                cuTENSOR.CUTENSOR_OP_MAX
        α = rand(eltyD)
        β = rand(eltyD)
        γ = rand(eltyD)
        dD = cuTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dC, indsC, opAB, opABC)
        D = collect(dD)
        if eltyD <: Complex
            if eltyA <: Complex && eltyB <: Complex
                @test D ≈ α .* permutedims(A, pA) .* β .* permutedims(B, pB) .+
                            γ .* conj.(C)
            elseif eltyB <: Complex
                @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, pA))) .*
                            β .* permutedims(B, pB) .+ γ .* conj.(C)
            elseif eltyB <: Complex
                @test D ≈ α .* permutedims(A, pA) .*
                            β .* sqrt.(convert.(eltyD, permutedims(B, pB))) .+
                            γ .* conj.(C)
            else
                @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, pA))) .*
                            β .* sqrt.(convert.(eltyD, permutedims(B, pB))) .+
                            γ .* conj.(C)
            end
        else
            @test D ≈ max.(min.(α .* sqrt.(convert.(eltyD, permutedims(A, pA))),
                                β .* sqrt.(convert.(eltyD, permutedims(B, pB)))),
                            γ .* C)
        end
    end
end
