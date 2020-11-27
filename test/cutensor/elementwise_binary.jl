using CUDA.CUTENSOR
using CUDA
using LinearAlgebra

# using host memory with CUTENSOR doesn't work on Windows
can_pin = !Sys.iswindows()

eltypes = ((Float16, Float16),
            #(Float16, Float32),
            (Float32, Float32),
            #(Float32, Float64),
            (Float64, Float64),
            #(ComplexF16, ComplexF16), (ComplexF16, ComplexF32),
            (ComplexF32, ComplexF32), #(ComplexF32, ComplexF64),
            (ComplexF64, ComplexF64))
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
        can_pin && Mem.pin(A)
        can_pin && Mem.pin(C)

        # simple case
        opA   = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC   = CUTENSOR.CUTENSOR_OP_IDENTITY
        dD    = similar(dC, eltyD)
        opAC  = CUTENSOR.CUTENSOR_OP_ADD
        dD    = CUTENSOR.elementwiseBinary!(1, dA, indsA, opA, 1, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C
        if can_pin
            Dsimple = similar(C)
             Mem.pin(Dsimple)
            Dsimple = CUDA.@sync CUTENSOR.elementwiseBinary!(1, A, indsA, opA, 1, C, indsC, opC,
                                                             Dsimple, indsC, opAC)
            @test Dsimple ≈ permutedims(A, p) .+ C
        end

        # using integers as indices
        dD = CUTENSOR.elementwiseBinary!(1, dA, 1:N, opA, 1, dC, p, opC, dD, p, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C
        if can_pin
            Dint = zeros(eltyC, dimsC...)
            Mem.pin(Dint)
            Dint = CUDA.@sync CUTENSOR.elementwiseBinary!(1, A, 1:N, opA, 1, C, p, opC, Dint, p, opAC)
            @test Dint ≈ permutedims(A, p) .+ C
        end

        # multiplication as binary operator
        opAC = CUTENSOR.CUTENSOR_OP_MUL
        dD = CUTENSOR.elementwiseBinary!(1, dA, indsA, opA, 1, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .* C
        if can_pin
            Dmult = zeros(eltyC, dimsC...)
            Mem.pin(Dmult)
            Dmult = CUDA.@sync CUTENSOR.elementwiseBinary!(1, A, indsA, opA, 1, C, indsC, opC,
                                                           Dmult, indsC, opAC)
            @test Dmult ≈ permutedims(A, p) .* C
        end

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opAC = CUTENSOR.CUTENSOR_OP_ADD
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = CUTENSOR.elementwiseBinary!(α, dA, indsA, opA, γ, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, p)) .+ γ .* C
        if can_pin
            Dnontrivial = similar(C)
            Mem.pin(Dnontrivial)
            Dnontrivial = CUDA.@sync CUTENSOR.elementwiseBinary!(α, A, indsA, opA, γ, C, indsC, opC,
                                                                 Dnontrivial, indsC, opAC)
            @test Dnontrivial ≈ α .* conj.(permutedims(A, p)) .+ γ .* C
        end

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                CUTENSOR.CUTENSOR_OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                CUTENSOR.CUTENSOR_OP_MAX
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = CUTENSOR.elementwiseBinary!(α, dA, indsA, opA, γ, dC, indsC, opC,
                                            dC, indsC, opAC)
        D = collect(dC)
        if eltyD <: Complex
            if eltyA <: Complex
                @test D ≈ α .* permutedims(A, p) .+ γ .* conj.(C)
            else
                @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, p))) .+
                            γ .* conj.(C)
            end
        else
            @test D ≈ max.(α .* sqrt.(convert.(eltyD, permutedims(A, p))), γ .* C)
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
        ctD = LinearAlgebra.axpy!(α, ctA, ctC)
        hD = collect(ctD.data)
        @test hD ≈ α.*permutedims(A, p) .+ C

        γ = rand(eltyD)
        ctD = LinearAlgebra.axpby!(α, ctA, γ, ctC)
        hD = collect(ctD.data)
        @test hD ≈ α.*permutedims(A, p) .+ γ.*C
    end
end
