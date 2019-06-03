@testset "CUTENSOR" begin

if !isdefined(CuArrays, :CUTENSOR)
@warn "Not testing CUTENSOR"
else
using CuArrays.CUTENSOR
@info "Testing CUTENSOR"

# const eltypes = (Float16, Float32, Float64, # ComplexF16,
#                     ComplexF32, ComplexF64)


@testset "Elementwise binary" begin
    eltypes = ((Float16, Float16), (Float16, Float32),
                            (Float32, Float32), (Float32, Float64),
                            (Float64, Float64),
                            #(ComplexF16, ComplexF16), (ComplexF16, ComplexF32),
                            (ComplexF32, ComplexF32), #(ComplexF32, ComplexF64),
                            (ComplexF64, ComplexF64))
    @testset for N=2:5
        @testset for (eltyA, eltyC) in eltypes
            # setup
            eltyD = eltyC
            dmax = 2^div(18,N)
            dims = rand(2:dmax, N)
            p = randperm(N)
            Ainds = collect(('a':'z')[1:N])
            Cinds = Ainds[p]
            dimsA = dims
            dimsC = dims[p]
            A = rand(eltyA, dimsA...)
            dA = CuArray(A)
            C = rand(eltyC, dimsC...)
            dC = CuArray(C)

            # simple case
            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            dD = similar(dC, eltyD)
            opAC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseBinary!(one(eltyD), dA, Ainds, Aop, one(eltyD), dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .+ C

            # using integers as indices
            dD = CUTENSOR.elementwiseBinary!(one(eltyD), dA, collect(1:N), Aop, one(eltyD), dC, p, Cop, dD, p, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .+ C

            # multiplication as binary operator
            opAC = CUTENSOR.CUTENSOR_OP_MUL
            dD = CUTENSOR.elementwiseBinary!(one(eltyD), dA, Ainds, Aop, one(eltyD), dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .* C

            # with non-trivial coefficients and conjugation
            Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opAC = CUTENSOR.CUTENSOR_OP_ADD
            α = rand(eltyD)
            β = rand(eltyD)
            dD = CUTENSOR.elementwiseBinary!(α, dA, Ainds, Aop, β, dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ α.*conj.(permutedims(A, p)) .+ β.*C

            # test in-place, and more complicated unary and binary operations
            Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                    CUTENSOR.CUTENSOR_OP_SQRT
            # because we use rand, entries of A will be positive when elty is real
            Cop = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            opAC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                    CUTENSOR.CUTENSOR_OP_MAX
            α = rand(eltyD)
            β = rand(eltyD)
            dD = CUTENSOR.elementwiseBinary!(α, dA, Ainds, Aop, β, dC, Cinds, Cop, dC, Cinds, opAC)
            D = collect(dC)
            if eltyD <: Complex
                if eltyA <: Complex
                    @test D ≈ α .* permutedims(A, p) .+ β .* conj.(C)
                else
                    @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, p))) .+
                                β .* conj.(C)
                end
            else
                @test D ≈ max.(α .* sqrt.(convert.(eltyD, permutedims(A, p))), β .* C)
            end            # # using host memory

            # using CuTensor type
            dA = CuArray(A)
            dC = CuArray(C)
            ctA = CuTensor(dA, Ainds)
            ctC = CuTensor(dC, Cinds)
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

            β = rand(eltyD)
            ctD = LinearAlgebra.axpby!(α, ctA, β, ctC)
            hD = collect(ctD.data)
            @test hD ≈ α.*permutedims(A, p) .+ β.*C
        end
    end
end

@testset "Permutations" begin
    eltypes = ((Float16, Float16), (Float16, Float32),
                            (Float32, Float32), (Float32, Float64),
                            (Float64, Float64),
                            #(ComplexF16, ComplexF16), (ComplexF16, ComplexF32),
                            (ComplexF32, ComplexF32), #(ComplexF32, ComplexF64),
                            (ComplexF64, ComplexF64))
    @testset for N=2:5
        @testset for (eltyA, eltyC) in eltypes
            # setup
            dmax = 2^div(18,N)
            dims = rand(2:dmax, N)
            p = randperm(N)
            Ainds = collect(('a':'z')[1:N])
            Cinds = Ainds[p]
            dimsA = dims
            dimsC = dims[p]
            A = rand(eltyA, dimsA...)
            dA = CuArray(A)
            dC = similar(dA, eltyC, dimsC...)

            # simple case
            dC = CUTENSOR.permutation!(one(eltyA), dA, Ainds, dC, Cinds)
            C  = collect(dC)
            @test C == permutedims(A, p) # exact equality

            # with scalar
            α = rand(eltyA)
            dC = CUTENSOR.permutation!(α, dA, Ainds, dC, Cinds)
            C  = collect(dC)
            @test C ≈ α * permutedims(A, p) # approximate, floating point rounding
        end
    end
end

@testset "Elementwise trinary" begin
    eltypes = ((Float16, Float16, Float16),
                (Float16, Float32, Float32),
                (Float32, Float16, Float32),
                (Float32, Float32, Float32),
                (Float32, Float32, Float64),
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
            Cinds = collect(('a':'z')[1:N])
            dimsC = dims
            Ainds = Cinds[ipA]
            dimsA = dims[ipA]
            Binds = Cinds[ipB]
            dimsB = dims[ipB]
            A = rand(eltyA, dimsA...)
            dA = CuArray(A)
            B = rand(eltyB, dimsB...)
            dB = CuArray(B)
            C = rand(eltyC, dimsC...)
            dC = CuArray(C)
            dD = similar(dC)

            # simple case
            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opAB = CUTENSOR.CUTENSOR_OP_ADD
            opABC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseTrinary!(1, dA, Ainds, Aop, 1, dB, Binds, Bop, 1, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C

            # using integers as indices
            dD = CUTENSOR.elementwiseTrinary!(1, dA, ipA, Aop, 1, dB, ipB, Bop, 1, dC, collect(1:N), Cop, dD, collect(1:N), opAB, opABC)
            D = collect(dD)
            @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C

            # multiplication as binary operator
            opAB = CUTENSOR.CUTENSOR_OP_MUL
            opABC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseTrinary!(1, dA, Ainds, Aop, 1, dB, Binds, Bop, 1, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ (convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB))) .+ C
            opAB = CUTENSOR.CUTENSOR_OP_ADD
            opABC = CUTENSOR.CUTENSOR_OP_MUL
            dD = CUTENSOR.elementwiseTrinary!(1, dA, Ainds, Aop, 1, dB, Binds, Bop, 1, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ (convert.(eltyD, permutedims(A, pA)) .+ convert.(eltyD, permutedims(B, pB))) .* C
            opAB = CUTENSOR.CUTENSOR_OP_MUL
            opABC = CUTENSOR.CUTENSOR_OP_MUL
            dD = CUTENSOR.elementwiseTrinary!(1, dA, Ainds, Aop, 1, dB, Binds, Bop, 1, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB)) .* C

            # with non-trivial coefficients and conjugation
            α = rand(eltyD)
            β = rand(eltyD)
            γ = rand(eltyD)
            Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            opAB = CUTENSOR.CUTENSOR_OP_ADD
            opABC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseTrinary!(α, dA, Ainds, Aop, β, dB, Binds, Bop, γ, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ α .* conj.(permutedims(A, pA)) .+ β .* permutedims(B, pB) .+ γ .* C

            Bop = eltyB <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            opAB = CUTENSOR.CUTENSOR_OP_ADD
            opABC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseTrinary!(α, dA, Ainds, Aop, β, dB, Binds, Bop, γ, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ α .* conj.(permutedims(A, pA)) .+ β .* conj.(permutedims(B, pB)) .+ γ .* C

            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opAB = CUTENSOR.CUTENSOR_OP_MUL
            opABC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseTrinary!(α, dA, Ainds, Aop, β, dB, Binds, Bop, γ, dC, Cinds, Cop, dD, Cinds, opAB, opABC)
            D = collect(dD)
            @test D ≈ α .* permutedims(A, pA) .* β .* conj.(permutedims(B, pB)) .+ γ .* C

            # test in-place, and more complicated unary and binary operations
            Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                    CUTENSOR.CUTENSOR_OP_SQRT
            Bop = eltyB <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                    CUTENSOR.CUTENSOR_OP_SQRT
            # because we use rand, entries of A will be positive when elty is real
            Cop = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            opAB = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_MUL :
                                    CUTENSOR.CUTENSOR_OP_MIN
            opABC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                    CUTENSOR.CUTENSOR_OP_MAX
            α = rand(eltyD)
            β = rand(eltyD)
            γ = rand(eltyD)
            dD = CUTENSOR.elementwiseTrinary!(α, dA, Ainds, Aop, β, dB, Binds, Bop, γ, dC, Cinds, Cop, dC, Cinds, opAB, opABC)
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
end

@testset "Contraction" begin
    eltypes = (# (Float16, Float16, Float16), # works for some
                # (Float16, Float16, Float32), # works for some but claims otherwise
                (Float32, Float32, Float32),
                # (Float32, Float32, Float64), # does not work
                (Float64, Float64, Float64),
                # (ComplexF16, ComplexF16, ComplexF16), # does not work
                # (ComplexF16, Complex ComplexF32), # does not work
                (ComplexF32, ComplexF32, ComplexF32),
                # (ComplexF32, ComplexF32, ComplexF64), # does not work
                (ComplexF64, ComplexF64, ComplexF64))

    @testset for NoA=1:3, NoB=1:3, Nc=1:3
        @testset for (eltyA, eltyB, eltyC) in eltypes
            # setup
            dmax = 2^div(18, max(NoA+Nc, NoB+Nc, NoA+NoB))
            dimsoA = rand(2:dmax, NoA)
            loA = prod(dimsoA)
            dimsoB = rand(2:dmax, NoB)
            loB = prod(dimsoB)
            dimsc = rand(2:dmax, Nc)
            lc = prod(dimsc)
            allinds = collect('a':'z')
            indsoA = allinds[1:NoA]
            indsoB = allinds[NoA .+ (1:NoB)]
            indsc = allinds[NoA .+ NoB .+ (1:Nc)]
            pA = randperm(NoA + Nc)
            ipA = invperm(pA)
            pB = randperm(Nc + NoB)
            ipB = invperm(pB)
            pC = randperm(NoA + NoB)
            ipC = invperm(pC)

            dimsA = [dimsoA; dimsc][pA]
            Ainds = [indsoA; indsc][pA]
            dimsB = [dimsc; dimsoB][pB]
            Binds = [indsc; indsoB][pB]
            dimsC = [dimsoA; dimsoB][pC]
            Cinds = [indsoA; indsoB][pC]

            A = rand(eltyA, dimsA...)
            mA = reshape(permutedims(A, ipA), (loA, lc))
            dA = CuArray(A)
            B = rand(eltyB, dimsB...)
            dB = CuArray(B)
            mB = reshape(permutedims(B, ipB), (lc, loB))
            C = zeros(eltyC, dimsC...)
            dC = CuArray(C)

            # simple case
            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
            dC = CUTENSOR.contraction!(1, dA, Ainds, Aop, dB, Binds, Bop, 0, dC, Cinds, Cop, opOut)
            C = collect(dC)
            mC = reshape(permutedims(C, ipC), (loA, loB))
            @test mC ≈ mA * mB

            # with non-trivial α
            α = rand(eltyC)
            dC = CUTENSOR.contraction!(α, dA, Ainds, Aop, dB, Binds, Bop, 0, dC, Cinds, Cop, opOut)
            C = collect(dC)
            mC = reshape(permutedims(C, ipC), (loA, loB))
            @test mC ≈ α * mA * mB

            # with non-trivial β
            C = rand(eltyC, dimsC...)
            dC = CuArray(C)
            α = rand(eltyC)
            β = rand(eltyC)
            copyto!(dC, C)
            dD = CUTENSOR.contraction!(α, dA, Ainds, Aop, dB, Binds, Bop, β, dC, Cinds, Cop, opOut)
            D = collect(dD)
            mC = reshape(permutedims(C, ipC), (loA, loB))
            mD = reshape(permutedims(D, ipC), (loA, loB))
            @test mD ≈ α * mA * mB + β * mC

            # with CuTensor objects
            ctA = CuTensor(dA, Ainds)
            ctB = CuTensor(dB, Binds)
            ctC = CuTensor(dC, Cinds)
            ctC = LinearAlgebra.mul!(ctC, ctA, ctB)
            C2, C2inds = collect(ctC)
            mC = reshape(permutedims(C2, ipC), (loA, loB))
            @test mC ≈ mA * mB
            ctC = ctA * ctB
            C2, C2inds = collect(ctC)
            pC2 = convert.(Int, indexin(convert.(Char, C2inds), [indsoA; indsoB]))
            mC = reshape(permutedims(C2, invperm(pC2)), (loA, loB))
            @test mC ≈ mA * mB

            # with conjugation flag for complex arguments
            if eltyA <: Complex
                Aop = CUTENSOR.CUTENSOR_OP_CONJ
                Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
                opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
                dC = CUTENSOR.contraction!(1, dA, Ainds, Aop, dB, Binds, Bop, 0, dC, Cinds, Cop, opOut)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ conj(mA) * mB
                # # not supported yet
                # opOut = CUTENSOR.CUTENSOR_OP_CONJ
                # dC = CUTENSOR.contraction!(α, dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
                # C = collect(dC)
                # mC = reshape(permutedims(C, ipC), (loA, loB))
                # @test mC ≈ conj(α * conj(mA) * mB)
            end
            if eltyB <: Complex
                Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
                Bop = CUTENSOR.CUTENSOR_OP_CONJ
                opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
                dC = CUTENSOR.contraction!(1, dA, Ainds, Aop, dB, Binds, Bop, 0, dC, Cinds, Cop, opOut)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ mA*conj(mB)
                # # not supported yet
                # opOut = CUTENSOR.CUTENSOR_OP_CONJ
                # dC = CUTENSOR.contraction!(α, dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
                # C = collect(dC)
                # mC = reshape(permutedims(C, ipC), (loA, loB))
                # @test mC ≈ conj(α * mA * conj(mB))
            end
            if eltyA <: Complex && eltyB <: Complex
                Aop = CUTENSOR.CUTENSOR_OP_CONJ
                Bop = CUTENSOR.CUTENSOR_OP_CONJ
                opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
                dC = CUTENSOR.contraction!(1, dA, Ainds, Aop, dB, Binds, Bop, 0, dC, Cinds, Cop, opOut)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ conj(mA)*conj(mB)
                # # not supported yet
                # opOut = CUTENSOR.CUTENSOR_OP_CONJ
                # dC = CUTENSOR.contraction!(α, dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
                # C = collect(dC)
                # mC = reshape(permutedims(C, ipC), (loA, loB))
                # @test mC ≈ conj(α * conj(mA) * conj(mB))
            end
        end
    end
end

end

end
