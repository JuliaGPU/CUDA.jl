@testset "applyMatrix! and expectation" begin
    # build a simple state and compute expectations
    @testset for elty in [ComplexF32, ComplexF64]
        result = (4.1, 0.0)
        h_sv   = elty[0.0, 0.1*im, 0.1+0.1im, 0.1+0.2im, 0.2+0.2im, 0.3+0.3im, 0.3+0.4im, 0.4+0.5im]
        O      = elty[1 2+im; 2-im 3]
        n_q    = 3
        sv     = CuStateVec(elty, n_q)
        copyto!(sv.data, h_sv)
        exp_res = expectation(sv, O, Int32[1])
        synchronize()
        @test exp_res[1][] ≈ result[1] atol=1e-6
        @test exp_res[2][] ≈ result[2]

        n_q = 2
        sv     = CuStateVec(elty, n_q)
        H  = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
        X  = convert(Matrix{elty}, [0 1; 1 0])
        Z  = convert(Matrix{elty}, [1 0; 0 -1])
        sv = CuStateVec(elty, n_q)
        sv = applyMatrix!(sv, X, false, Int32[0], Int32[])
        sv = applyMatrix!(sv, X, false, Int32[1], Int32[])
        exp, res = expectation(sv, CuMatrix(Z), Int32[0])
        synchronize()
        @test exp[] ≈ -1.0 atol=1e-6
        exp, res = expectation(sv, CuMatrix(Z), Int32[1])
        @test exp[] ≈ -1.0 atol=1e-6
        exp, res = expectation(sv, CuMatrix(X), Int32[0])
        @test exp[] ≈ 0.0 atol=1e-6
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
        synchronize()
        @test exp[] ≈ 0.0 atol=1e-6
        exp, res = expectation(sv, X, Int32[0])
        synchronize()
        @test exp[] ≈ 0.0 atol=1e-6
    end
    # with expectationsOnPauliBasis
    n_q = 2
    @testset for elty in [ComplexF32, ComplexF64]
        H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
        sv = CuStateVec(elty, n_q)
        sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
        sv = applyMatrix!(sv, H, false, Int32[1], Int32[])
        pauli_ops = [cuStateVec.Pauli[cuStateVec.PauliX()], cuStateVec.Pauli[cuStateVec.PauliX()]]
        exp_vals = expectationsOnPauliBasis(sv, pauli_ops, [[0], [1]])
        @test exp_vals[1] ≈ 1.0 atol=1e-6
        @test exp_vals[2] ≈ 1.0 atol=1e-6


        H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
        sv = CuStateVec(elty, n_q)
        sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
        sv = applyMatrix!(sv, H, false, Int32[1], Int32[])
        pauli_ops = [cuStateVec.Pauli[cuStateVec.PauliY()], cuStateVec.Pauli[cuStateVec.PauliI()]]
        exp_vals = expectationsOnPauliBasis(sv, pauli_ops, [[0], [1]])
        @test exp_vals[1] ≈ 0.0 atol=1e-6
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
                batched_vec = zeros(elty, n_svs*2^(n_q))
                for sv_ix in 0:n_svs-1
                    batched_vec[sv_ix*(2^n_q) + 1] = one(elty)
                end
                sv = CuStateVec(elty, n_svs * n_q) # padded state vector
                copyto!(sv.data, batched_vec)
                H_batch = CuVector{elty}(repeat(vec(H), n_mats))
                sv = applyMatrixBatched!(sv, n_svs, mapping, mat_inds, H_batch, n_mats, false, Int32[0], Int32[])
                CUDA.@allowscalar begin
                    for sv_ix in 0:n_svs-1
                        ix_begin = sv_ix*2^n_q + 1
                        ix_end   = (sv_ix+1)*2^n_q
                        sv_ = CuStateVec(elty, n_q)
                        sv_.data .= sv.data[ix_begin:ix_end]
                        exp, res = expectation(sv_, Z, Int32[0])
                        synchronize()
                        @test exp[] ≈ 0.0 atol=1e-6
                        exp, res = expectation(sv_, X, Int32[0])
                        synchronize()
                        @test exp[] ≈ 1.0 atol=1e-6
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
        synchronize()
        @test exp[] ≈ 0.0 atol=1e-6
        exp, res = expectation(sv, X, Int32[0])
        synchronize()
        @test exp[] ≈ 0.0 atol=1e-6
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
