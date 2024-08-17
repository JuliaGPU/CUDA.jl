using Test, LinearAlgebra

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using cuStateVec
@test cuStateVec.has_custatevec()
@info "cuStateVec version: $(cuStateVec.version())"

@testset "cuStateVec" begin
    import cuStateVec: CuStateVec, applyMatrix!, applyMatrixBatched!, applyPauliExp!, applyGeneralizedPermutationMatrix!, expectation, expectationsOnPauliBasis, sample, testMatrixType, Pauli, PauliX, PauliY, PauliZ, PauliI, measureOnZBasis!, swapIndexBits!, abs2SumOnZBasis, collapseOnZBasis!, batchMeasure!, abs2SumArray, collapseByBitString!, abs2SumArrayBatched, collapseByBitStringBatched!

    @testset "applyMatrix! and expectation" begin
        # build a simple state and compute expectations
        n_q = 2
        @testset for elty in [ComplexF32, ComplexF64]
            H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
            X = convert(Matrix{elty}, [0 1; 1 0])
            Z = convert(Matrix{elty}, [1 0; 0 -1])
            sv = CuStateVec(elty, n_q)
            sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
            sv = applyMatrix!(sv, H, false, Int32[1], Int32[])
            exp, res = expectation(sv, Z, Int32[1])
            @test exp ≈ 0.0 atol=1e-6
            exp, res = expectation(sv, X, Int32[0])
            @test exp ≈ 1.0 atol=1e-6
        end
        # build a simple state with controls and compute expectations
        n_q = 2
        @testset for elty in [ComplexF32, ComplexF64]
            H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
            X = convert(Matrix{elty}, [0 1; 1 0])
            Z = convert(Matrix{elty}, [1 0; 0 -1])
            sv = CuStateVec(elty, n_q)
            sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
            sv = applyMatrix!(sv, X, false, Int32[1], Int32[0]) # CNOT
            exp, res = expectation(sv, Z, Int32[0])
            @test exp ≈ 0.0 atol=1e-6
            exp, res = expectation(sv, X, Int32[0])
            @test exp ≈ 0.0 atol=1e-6
        end
        # with expectationsOnPauliBasis
        n_q = 2
        @testset for elty in [ComplexF32, ComplexF64]
            H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
            X = convert(Matrix{elty}, [0 1; 1 0])
            Z = convert(Matrix{elty}, [1 0; 0 -1])
            sv = CuStateVec(elty, n_q)
            sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
            sv = applyMatrix!(sv, H, false, Int32[1], Int32[])
            pauli_ops = [cuStateVec.Pauli[cuStateVec.PauliX()], cuStateVec.Pauli[cuStateVec.PauliX()]]
            exp_vals = expectationsOnPauliBasis(sv, pauli_ops, [[0], [1]])
            @test exp_vals[1] ≈ 1.0 atol=1e-6
            @test exp_vals[2] ≈ 1.0 atol=1e-6
        end
    end
    @testset "applyMatrixBatched! and expectation" begin
        # build a simple state and compute expectations
        n_q = 2
        @testset for elty in [ComplexF32, ComplexF64]
            H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
            X = convert(Matrix{elty}, [0 1; 1 0])
            Z = convert(Matrix{elty}, [1 0; 0 -1])
            @testset for n_svs in (1, 2)
                @testset for (mapping, mat_inds, n_mats) in (
                                                             (cuStateVec.CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, collect(0:n_svs-1), n_svs), 
                                                             (cuStateVec.CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, fill(0, n_svs), 1), 
                                                             (cuStateVec.CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, fill(0, n_svs), 1),
                                                  )
                    batched_vec = CUDA.zeros(elty, n_svs*2^(n_q))
                    for sv_ix in 0:n_svs-1
                        CUDA.@allowscalar batched_vec[sv_ix*(2^n_q) + 1] = one(elty)
                    end
                    sv = CuStateVec(batched_vec) # padded state vector
                    H_batch = CuVector{elty}(repeat(vec(H), n_mats))
                    sv = applyMatrixBatched!(sv, n_svs, mapping, mat_inds, H_batch, n_mats, false, Int32[0], Int32[])
                    CUDA.@allowscalar begin
                        for sv_ix in 0:n_svs-1
                            ix_begin = sv_ix*2^n_q + 1
                            ix_end   = (sv_ix+1)*2^n_q
                            sv_ = CuStateVec(sv.data[ix_begin:ix_end])
                            exp, res = expectation(sv_, Z, Int32[0])
                            @test exp ≈ 0.0 atol=1e-6
                            exp, res = expectation(sv_, X, Int32[0])
                            @test exp ≈ 1.0 atol=1e-6
                        end
                    end
                end
            end
        end
        # build a simple state with controls and compute expectations
        n_q = 2
        @testset for elty in [ComplexF32, ComplexF64]
            H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
            X = convert(Matrix{elty}, [0 1; 1 0])
            Z = convert(Matrix{elty}, [1 0; 0 -1])
            sv = CuStateVec(elty, n_q)
            sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
            sv = applyMatrix!(sv, X, false, Int32[1], Int32[0]) # CNOT
            exp, res = expectation(sv, Z, Int32[0])
            @test exp ≈ 0.0 atol=1e-6
            exp, res = expectation(sv, X, Int32[0])
            @test exp ≈ 0.0 atol=1e-6
        end
    end
    @testset "applyMatrix! and sample" begin
        # build a simple state and compute samples
        n_q = 10
        @testset for elty in [ComplexF32, ComplexF64]
            H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
            sv = CuStateVec(elty, n_q)
            for q in 0:n_q-1
                sv = applyMatrix!(sv, H, false, Int32[q], Int32[])
            end
            samples = sample(sv, Int32.(collect(0:n_q-1)), 10)
            # check that this succeeded
            @test length(samples) == 10
        end
    end
    @testset "applyPauliExp" begin
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = 1.0/√8 .* elty[0.0, im, 0.0, im, 0.0, im, 0.0, im]
            h_sv_result = 1.0/√8 * elty[0.0, im, 0.0, -1.0, 0.0, im, 0.0, 1.0]
            sv   = CuStateVec(h_sv)
            sv   = applyPauliExp!(sv, π/2, [PauliZ()], Int32[2], Int32[1], Int32[1])
            sv_result = collect(sv.data)
            @test sv_result ≈ h_sv_result
        end
    end
    @testset "applyGeneralizedPermutationMatrix" begin
        @testset for elty in [ComplexF32, ComplexF64]
            diagonals   = elty[1.0, im, im, 1.0]
            permutation = [0, 2, 1, 3]
            h_sv        = elty[0.0, 0.1im, 0.1 + 0.1im, 0.1 + 0.2im, 0.2+0.2im, 0.3 + 0.3im, 0.3+0.4im, 0.4+0.5im]
            h_sv_result = elty[0.0, 0.1im, 0.1 + 0.1im, 0.1 + 0.2im, 0.2+0.2im, -0.4 + 0.3im, -0.3+0.3im, 0.4+0.5im]
            sv = CuStateVec{elty}(CuVector{elty}(h_sv), UInt32(log2(length(h_sv))))
            sv_result = applyGeneralizedPermutationMatrix!(sv, CuVector{Int64}(permutation), CuVector{elty}(diagonals), false, [0, 1], [2], [1])
            @test collect(sv_result.data) ≈ h_sv_result
        end
    end
    @testset "abs2sumOnZBasis and collapseOnZBasis!" begin
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = 1.0/√8 .* elty[0.0, im, 0.0, im, 0.0, im, 0.0, im]
            h_sv_result_0 = 1.0/√2 * elty[0.0, 0.0, 0.0, im, 0.0, im,  0.0, 0.0]
            h_sv_result_1 = 1.0/√2 * elty[0.0, im, 0.0, 0.0, 0.0, 0.0, 0.0, im]
            sv   = CuStateVec(h_sv)
            abs2sum0, abs2sum1 = abs2SumOnZBasis(sv, [0, 1, 2])
            abs2sum = abs2sum0 + abs2sum1
            for (parity, norm, h_sv_result) in ((0, abs2sum0, h_sv_result_0), (1, abs2sum1, h_sv_result_1))
                d_sv = copy(sv)
                d_sv = collapseOnZBasis!(d_sv, parity, [0, 1, 2], norm)
                sv_result  = collect(d_sv.data)
                @test sv_result ≈ h_sv_result
            end
        end
    end
    @testset "measureOnZBasis" begin
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = 1.0/√8 .* elty[0.0, im, 0.0, im, 0.0, im, 0.0, im]
            h_sv_result = 1.0/√2 * elty[0.0, 0.0, 0.0, im, 0.0, 0.0, 0.0, im]
            sv   = CuStateVec(h_sv)
            sv, parity = measureOnZBasis!(sv, [0, 1, 2], 0.2, cuStateVec.CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO)
            sv_result  = collect(sv.data)
            @test sv_result ≈ h_sv_result
        end
    end
    @testset "abs2SumArray and collapseByBitString!" begin
        nq = 3
        bit_ordering = [2, 1, 0]
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = elty[0.0, 0.1*im, 0.1+0.1*im, 0.1+0.2*im, 0.2+0.2*im, 0.3+0.3im, 0.3+0.4*im, 0.4+0.5*im]
            h_sv_result = elty[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3+0.4*im, 0.0]
            sv   = CuStateVec(h_sv)
            abs2sum = abs2SumArray(sv, bit_ordering, Int[], Int[])
            bitstr = [1, 1, 0]
            d_sv = copy(sv)
            d_sv = collapseByBitString!(d_sv, bitstr, bit_ordering, 1.)
            sv_result  = collect(d_sv.data)
            @test sv_result ≈ h_sv_result
        end
    end
    @testset "abs2SumArrayBatched" begin
        bit_ordering = [1]
        @testset for elty in [ComplexF32, ComplexF64]
            @testset for n_svs in (2,)
                h_sv = elty[0.0, 0.1*im, 0.1 + 0.1*im, 0.1 + 0.2*im, 0.2+0.2*im, 0.3+0.3*im, 0.3+0.4*im, 0.4+0.5*im, 0.25+0.25*im, 0.25+0.25*im, 0.25+0.25*im, 0.25+0.25*im, 0.25+0.25*im, 0.25+0.25*im, 0.25+0.25*im, 0.25+0.25*im]
                a2s_result = real(elty)[0.27, 0.73, 0.5, 0.5]
                sv      = CuStateVec(h_sv)
                abs2sum = abs2SumArrayBatched(sv, n_svs, bit_ordering, Int[], Int[])
                @test abs2sum ≈ a2s_result
            end
        end
    end
    @testset "collapseByBitStringBatched!" begin
        bit_ordering = [0, 1, 2]
        @testset for elty in [ComplexF32, ComplexF64]
            @testset for n_svs in (2,)
                h_sv = elty[0.0, 0.1*im, 0.1 + 0.1*im, 0.1 + 0.2*im, 0.2+0.2*im, 0.3+0.3*im, 0.3+0.4*im, 0.4+0.5*im, 0.0, 0.1*im, 0.1+0.1*im, 0.1+0.2*im, 0.2+0.2*im, 0.3+0.3*im, 0.3+0.4*im, 0.4*0.5*im]
                h_sv_result = elty[0.0, im, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6+0.8*im, 0.0]
                sv      = CuStateVec(h_sv)
                bitstr = [0b001, 0b110]
                d_sv = copy(sv)
                d_sv = collapseByBitStringBatched!(d_sv, n_svs, bitstr, bit_ordering, [0.01, 0.25])
                sv_result  = collect(d_sv.data)
                @test sv_result ≈ h_sv_result
            end
        end
    end
    @testset "batchMeasure!" begin
        nq = 3
        bit_ordering = [2, 1, 0]
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = elty[0.0, 0.1*im, 0.1+0.1*im, 0.1+0.2*im, 0.2+0.2*im, 0.3+0.3im, 0.3+0.4*im, 0.4+0.5*im]
            h_sv_result = elty[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6+0.8*im, 0.0]
            sv   = CuStateVec(h_sv)
            sv, bitstr = batchMeasure!(sv, bit_ordering, 0.5, cuStateVec.CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO)
            sv_result  = collect(sv.data)
            @test sv_result ≈ h_sv_result
            @test bitstr == [1, 1, 0]
        end
    end
    @testset "swapIndexBits" begin
        @testset for elty in [ComplexF32, ComplexF64]
            # 0.1|000> + 0.4|011> - 0.4|101> - 0.3im|111>
            h_sv = elty[0.1, 0.0, 0.0, 0.4, 0.0, -0.4, 0.0, -0.3im]
            # 0.1|000> + 0.4|110> - 0.4|101> + 0.3im|111>
            h_sv_result = elty[0.1, 0.0, 0.0, 0.0, 0.0, -0.4, 0.4, -0.3im]
            sv   = CuStateVec(h_sv)
            swapped_sv = swapIndexBits!(sv, [0=>2], [1], [1])
            @test collect(swapped_sv.data) == h_sv_result
        end
    end
    @testset "testMatrixType $elty" for elty in [ComplexF32, ComplexF64]
        n = 128
        @testset "Hermitian matrix" begin
            A = rand(elty, n, n)
            A = A + A'
            @test testMatrixType(A, false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
            @test testMatrixType(A, true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
        end
        @testset "Unitary matrix" begin
            A = elty <: Real ? diagm(ones(elty, n)) : exp(im * 0.2 * diagm(ones(elty, n)))
            @test testMatrixType(A, false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
            @test testMatrixType(A, true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
        end
    end
end
