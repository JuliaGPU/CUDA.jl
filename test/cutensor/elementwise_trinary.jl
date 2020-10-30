using CUDA.CUTENSOR
using CUDA
using LinearAlgebra

# using host memory with CUTENSOR doesn't work on Windows
can_pin = !Sys.iswindows()

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
        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opB = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C

        # test with Mem.pinned host memory
        if can_pin
            Mem.pin(A)
            Mem.pin(B)
            Mem.pin(C)
            Dc = similar(C)
            Mem.pin(Dc)
            Dc = CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                1, C, indsC, opC, Dc, indsC, opAB, opABC)
            synchronize()
            @test Dc ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C
        end

        # using integers as indices
        dD = CUTENSOR.elementwiseTrinary!(1, dA, ipA, opA, 1, dB, ipB, opB,
                                            1, dC, 1:N, opC, dD, 1:N, opAB, opABC)
        D = collect(dD)
        @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C
        if can_pin
            Dd = similar(C)
            Mem.pin(Dd)
            Dd = CUTENSOR.elementwiseTrinary!(1, A, ipA, opA, 1, B, ipB, opB,
                                                1, C, 1:N, opC, Dd, 1:N, opAB, opABC)
            synchronize()
            @test Dd ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C
        end

        # multiplication as binary operator
        opAB = CUTENSOR.CUTENSOR_OP_MUL
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ (convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB))) .+ C
        if can_pin
            De = similar(C)
            Mem.pin(De)
            De = CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                1, C, indsC, opC, De, indsC, opAB, opABC)
            synchronize()
            @test De ≈ (convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB))) .+ C
        end

        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_MUL
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ (convert.(eltyD, permutedims(A, pA)) .+ convert.(eltyD, permutedims(B, pB))) .* C
        if can_pin
            Df = similar(C)
            Mem.pin(Df)
            Df = CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                1, C, indsC, opC, Df, indsC, opAB, opABC)
            synchronize()
            @test Df ≈ (convert.(eltyD, permutedims(A, pA)) .+ convert.(eltyD, permutedims(B, pB))) .* C
        end

        opAB = CUTENSOR.CUTENSOR_OP_MUL
        opABC = CUTENSOR.CUTENSOR_OP_MUL
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ convert.(eltyD, permutedims(A, pA)) .*
                    convert.(eltyD, permutedims(B, pB)) .* C
        if can_pin
            Dg = similar(C)
            Mem.pin(Dg)
            Dg = CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                1, C, indsC, opC, Dg, indsC, opAB, opABC)
            synchronize()
            @test Dg ≈ convert.(eltyD, permutedims(A, pA)) .*
                        convert.(eltyD, permutedims(B, pB)) .* C
        end

        # with non-trivial coefficients and conjugation
        α = rand(eltyD)
        β = rand(eltyD)
        γ = rand(eltyD)
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, pA)) .+ β .* permutedims(B, pB) .+ γ .* C
        if can_pin
            Dh = similar(C)
            Mem.pin(Dh)
            Dh = CUTENSOR.elementwiseTrinary!(α, A, indsA, opA, β, B, indsB, opB,
                                                γ, C, indsC, opC, Dh, indsC, opAB, opABC)
            synchronize()
            @test Dh ≈ α .* conj.(permutedims(A, pA)) .+ β .* permutedims(B, pB) .+ γ .* C
        end

        opB = eltyB <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, pA)) .+
                    β .* conj.(permutedims(B, pB)) .+ γ .* C
        if can_pin
            Di = similar(C)
            Mem.pin(Di)
            Di = CUTENSOR.elementwiseTrinary!(α, A, indsA, opA, β, B, indsB, opB,
                                                γ, C, indsC, opC, Di, indsC, opAB, opABC)
            synchronize()
            @test Di ≈ α .* conj.(permutedims(A, pA)) .+
                        β .* conj.(permutedims(B, pB)) .+ γ .* C
        end

        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_MUL
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* permutedims(A, pA) .* β .* conj.(permutedims(B, pB)) .+ γ .* C
        if can_pin
            Dj = similar(C)
            Mem.pin(Dj)
            Dj = CUTENSOR.elementwiseTrinary!(α, A, indsA, opA, β, B, indsB, opB,
                                                γ, C, indsC, opC, Dj, indsC, opAB, opABC)
            synchronize()
            @test Dj ≈ α .* permutedims(A, pA) .* β .* conj.(permutedims(B, pB)) .+ γ .* C
        end

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                CUTENSOR.CUTENSOR_OP_SQRT
        opB = eltyB <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                CUTENSOR.CUTENSOR_OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_MUL :
                                CUTENSOR.CUTENSOR_OP_MIN
        opABC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                CUTENSOR.CUTENSOR_OP_MAX
        α = rand(eltyD)
        β = rand(eltyD)
        γ = rand(eltyD)
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
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
