using CUDA.CUTENSOR
using CUDA
using LinearAlgebra

eltypes = (#(Float16, Float16), #(Float16, Float32),
                        (Float32, Float32), #(Float32, Float64),
                        (Float64, Float64),
                        #(ComplexF16, ComplexF16), (ComplexF16, ComplexF32),
                        (ComplexF32, ComplexF32), #(ComplexF32, ComplexF64),
                        (ComplexF64, ComplexF64))
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
        # setup
        Mem.pin(A)

        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opReduce = CUTENSOR.CUTENSOR_OP_ADD
        # simple case
        dC = CUTENSOR.reduction!(1, dA, indsA, opA, 0, dC, indsC, opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))
        Csimple = zeros(eltyC, dimsC...)
        Mem.pin(Csimple)
        Csimple = CUTENSOR.reduction!(1, A, indsA, opA, 0, Csimple, indsC, opC, opReduce)
        synchronize()
        @test reshape(Csimple, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))

        # using integers as indices
        Cinteger = zeros(eltyC, dimsC...)
        Mem.pin(Cinteger)
        dC = CUTENSOR.reduction!(1, dA, collect(1:NA), opA, 0, dC, p[1:NC], opC, opReduce)
        synchronize()
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))
        Cinteger = CUTENSOR.reduction!(1, A, collect(1:NA), opA, 0, Cinteger, p[1:NC], opC, opReduce)
        synchronize()
        @test reshape(Cinteger, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))

        # multiplication as reduction operator
        opReduce = CUTENSOR.CUTENSOR_OP_MUL
        Cmult = zeros(eltyC, dimsC...)
        Mem.pin(Cmult)
        dC = CUTENSOR.reduction!(1, dA, indsA, opA, 0, dC, indsC, opC, opReduce)
        synchronize()
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            prod(permutedims(A, p); dims = ((NC+1:NA)...,)) atol=eps(Float16) rtol=Base.rtoldefault(Float16)
        Cmult = CUTENSOR.reduction!(1, A, indsA, opA, 0, Cmult, indsC, opC, opReduce)
        synchronize()
        @test reshape(Cmult, (dimsC..., ones(Int,NA-NC)...)) ≈
            prod(permutedims(A, p); dims = ((NC+1:NA)...,)) atol=eps(Float16) rtol=Base.rtoldefault(Float16)
        # NOTE: this test often yields values close to 0 that do not compare approximately

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opReduce = CUTENSOR.CUTENSOR_OP_ADD
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)
        α = rand(eltyC)
        γ = rand(eltyC)
        dC = CUTENSOR.reduction!(α, dA, indsA, opA, γ, dC, indsC, opC, opReduce)
        @test reshape(collect(dC), (dimsC..., ones(Int,NA-NC)...)) ≈
            α .* conj.(sum(permutedims(A, p); dims = ((NC+1:NA)...,))) .+ γ .* C
    end
end
