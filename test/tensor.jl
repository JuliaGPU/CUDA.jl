@testset "CUTENSOR" begin

if !isdefined(CuArrays, :CUTENSOR)
@warn "Not testing CUTENSOR"
else
using CuArrays.CUTENSOR
@info "Testing CUTENSOR"

@testset "Elementwise binary" begin
    eltypes = (Float16, Float32, Float64, # ComplexF16,
                ComplexF32, ComplexF64)
    @testset for N=2:5
        @testset for elty in eltypes
            eltyA = elty
            eltyC = elty
        # @testset for (eltyA, eltyC) in Base.Iterators.product(eltypes, eltypes)
        # mixed precision does not seem to work currently
            eltyD = promote_type(eltyA, eltyC)
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
            # simple
            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            dD = similar(dC, eltyD)
            opAC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseBinary!(one(eltyA), dA, Ainds, Aop, one(eltyC), dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .+ C
            # using integers as indices
            dD = CUTENSOR.elementwiseBinary!(one(eltyA), dA, collect(1:N), Aop, one(eltyC), dC, p, Cop, dD, p, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .+ C
            # multiplication as binary operator
            opAC = CUTENSOR.CUTENSOR_OP_MUL
            dD = CUTENSOR.elementwiseBinary!(one(eltyA), dA, Ainds, Aop, one(eltyC), dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .* C
            # with non-trivial coefficients
            Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opAC = CUTENSOR.CUTENSOR_OP_ADD
            α = rand(eltyA)
            β = rand(eltyC)
            dD = CUTENSOR.elementwiseBinary!(α, dA, Ainds, Aop, β, dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ α.*conj.(permutedims(A, p)) .+ β.*C
            # test in-place, and more complicated unary and binary operations
            if eltyD == eltyC
                Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                        CUTENSOR.CUTENSOR_OP_SQRT
                # because we use rand, entries of A will be positive when elty is real
                Cop = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                        CUTENSOR.CUTENSOR_OP_IDENTITY
                opAC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                        CUTENSOR.CUTENSOR_OP_MAX
                α = rand(eltyA)
                β = rand(eltyC)
                dD = CUTENSOR.elementwiseBinary!(α, dA, Ainds, Aop, β, dC, Cinds, Cop, dC, Cinds, opAC)
                D = collect(dC)
                if eltyD <: Complex
                    if eltyA <: Complex
                        @test D ≈ α.*permutedims(A, p) .+ β.*conj.(C)
                    else
                        @test D ≈ α.*sqrt.(permutedims(A, p)) .+ β.*conj.(C)
                    end
                else
                    @test D ≈ max.(α.*sqrt.(permutedims(A, p)), β.*C)
                end
            end

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

            α = rand(eltyA)
            ctD = LinearAlgebra.axpy!(α, ctA, ctC)
            hD = collect(ctD.data)
            @test hD ≈ α.*permutedims(A, p) .+ C

            β = rand(eltyA)
            ctD = LinearAlgebra.axpby!(α, ctA, β, ctC)
            hD = collect(ctD.data)
            @test hD ≈ α.*permutedims(A, p) .+ β.*C
        end
    end
end

#=@testset "Elementwise trinary" begin
    eltyA       = Float32
    eltyB       = Float32
    eltyC       = Float32
    eltyCompute = Float32
    Ainds = ['c', 'b', 'a']
    Binds = ['c', 'a', 'b']
    Cinds = ['a', 'b', 'c']
    a_size = 30
    b_size = 20
    c_size = 40
    dimsA = (c_size, b_size, a_size)
    dimsB = (c_size, a_size, b_size)
    dimsC = (a_size, b_size, c_size)
    A = rand(eltyA, dimsA...)
    dA = CuArray(A)
    B = rand(eltyB, dimsB...)
    dB = CuArray(B)
    C = rand(eltyC, dimsC...)
    dC = CuArray(C)
    Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
    dD = similar(dC)
    opAB = CUTENSOR.CUTENSOR_OP_ADD
    opABC = CUTENSOR.CUTENSOR_OP_ADD
    dD = CUTENSOR.elementwiseTrinary!(Float32(1.0), dA, Ainds, Aop, Float32(1.0), dB, Binds, Bop, Float32(1.0), dC, Cinds, Cop, dD, Cinds, opAB, opABC)
    D = collect(dD)
    @test D ≈ permutedims(A, [3, 2, 1]) + permutedims(B, [2, 3, 1]) + C

    #=A = rand(eltyA, dimsA...)
    B = rand(eltyB, dimsB...)
    C = rand(eltyC, dimsC...)
    Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
    opAB = CUTENSOR.CUTENSOR_OP_ADD
    opABC = CUTENSOR.CUTENSOR_OP_ADD
    D = CUTENSOR.elementwiseTrinary!(Float32(1.0), A, Ainds, Aop, Float32(1.0), B, Binds, Bop, Float32(1.0), C, Cinds, Cop, D, Cinds, opAB, opABC)
    @test D ≈ permutedims(A, [3, 2, 1]) + permutedims(B, [2, 3, 1]) + C=#
end=#

@testset "Permutations" begin
    eltypes = (Float16, Float32, Float64, # ComplexF16,
                ComplexF32, ComplexF64)
    @testset for N=2:5
        @testset for elty in eltypes
            eltyA = elty
            eltyC = elty
        # @testset for (eltyA, eltyC) in Base.Iterators.product(eltypes, eltypes)
        # mixed precision does not seem to work currently
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

@testset "Contraction" begin
    eltypes = (Float32, Float64, # Float16, ComplexF16,
                ComplexF32, ComplexF64)
    @testset for NoA=1:3, NoB=1:3, Nc=1:3
        # same eltype for both arguments
        @testset for elty in eltypes
            eltyA = elty
            eltyB = elty
        # @testset for (eltyA, eltyB) in Base.Iterators.product(eltypes, eltypes)
        # mixed precision does not seem to work currently
            eltyC = promote_type(eltyA, eltyB)

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
            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
            dC = CUTENSOR.contraction!(one(eltyA), dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
            C = collect(dC)
            mC = reshape(permutedims(C, ipC), (loA, loB))
            @test mC ≈ mA * mB
            # with non-trivial α
            α = rand(eltyC)
            dC = CUTENSOR.contraction!(α, dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
            C = collect(dC)
            mC = reshape(permutedims(C, ipC), (loA, loB))
            @test mC ≈ α * mA * mB
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
                dC = CUTENSOR.contraction!(one(eltyA), dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
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
                dC = CUTENSOR.contraction!(one(eltyA), dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
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
                dC = CUTENSOR.contraction!(one(eltyA), dA, Ainds, Aop, dB, Binds, Bop, zero(eltyC), dC, Cinds, Cop, opOut)
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
