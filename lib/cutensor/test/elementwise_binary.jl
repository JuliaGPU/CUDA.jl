using CUDA, cuTENSOR
using LinearAlgebra

# using host memory with cuTENSOR doesn't work on Windows
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
        opA   = cuTENSOR.CUTENSOR_OP_IDENTITY
        opC   = cuTENSOR.CUTENSOR_OP_IDENTITY
        dD    = similar(dC, eltyD)
        opAC  = cuTENSOR.CUTENSOR_OP_ADD
        dD    = cuTENSOR.elementwiseBinary!(1, dA, indsA, opA, 1, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C
        if can_pin
            Dsimple = similar(C)
             Mem.pin(Dsimple)
            Dsimple = CUDA.@sync cuTENSOR.elementwiseBinary!(1, A, indsA, opA, 1, C, indsC, opC,
                                                             Dsimple, indsC, opAC)
            @test Dsimple ≈ permutedims(A, p) .+ C
        end

        # using integers as indices
        dD = cuTENSOR.elementwiseBinary!(1, dA, 1:N, opA, 1, dC, p, opC, dD, p, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C
        if can_pin
            Dint = zeros(eltyC, dimsC...)
            Mem.pin(Dint)
            Dint = CUDA.@sync cuTENSOR.elementwiseBinary!(1, A, 1:N, opA, 1, C, p, opC, Dint, p, opAC)
            @test Dint ≈ permutedims(A, p) .+ C
        end

        # multiplication as binary operator
        opAC = cuTENSOR.CUTENSOR_OP_MUL
        dD = cuTENSOR.elementwiseBinary!(1, dA, indsA, opA, 1, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .* C
        if can_pin
            Dmult = zeros(eltyC, dimsC...)
            Mem.pin(Dmult)
            Dmult = CUDA.@sync cuTENSOR.elementwiseBinary!(1, A, indsA, opA, 1, C, indsC, opC,
                                                           Dmult, indsC, opAC)
            @test Dmult ≈ permutedims(A, p) .* C
        end

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? cuTENSOR.CUTENSOR_OP_CONJ :
                                cuTENSOR.CUTENSOR_OP_IDENTITY
        opC = cuTENSOR.CUTENSOR_OP_IDENTITY
        opAC = cuTENSOR.CUTENSOR_OP_ADD
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = cuTENSOR.elementwiseBinary!(α, dA, indsA, opA, γ, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, p)) .+ γ .* C
        if can_pin
            Dnontrivial = similar(C)
            Mem.pin(Dnontrivial)
            Dnontrivial = CUDA.@sync cuTENSOR.elementwiseBinary!(α, A, indsA, opA, γ, C, indsC, opC,
                                                                 Dnontrivial, indsC, opAC)
            @test Dnontrivial ≈ α .* conj.(permutedims(A, p)) .+ γ .* C
        end

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? cuTENSOR.CUTENSOR_OP_IDENTITY :
                                cuTENSOR.CUTENSOR_OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? cuTENSOR.CUTENSOR_OP_CONJ :
                                cuTENSOR.CUTENSOR_OP_IDENTITY
        opAC = eltyD <: Complex ? cuTENSOR.CUTENSOR_OP_ADD :
                                cuTENSOR.CUTENSOR_OP_MAX
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = cuTENSOR.elementwiseBinary!(α, dA, indsA, opA, γ, dC, indsC, opC,
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
