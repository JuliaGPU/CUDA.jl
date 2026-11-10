@testset "deprecated names" begin
    @testset for elty in [ComplexF32, ComplexF64]
        sv = @test_deprecated CuStateVec(elty, 2)
        @test sv isa CuStateVec{elty}
        @test collect(sv.data) == elty[1, 0, 0, 0]
        sv = @test_deprecated CuStateVec(elty, 2; sv_type=cuStateVec.CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM)
        @test collect(sv.data) ≈ fill(elty(0.5), 4)

        h_sv = 1.0/√8 .* elty[0.0, im, 0.0, im, 0.0, im, 0.0, im]
        h_sv_result = 1.0/√8 * elty[0.0, im, 0.0, -1.0, 0.0, im, 0.0, 1.0]
        sv = @test_deprecated cuStateVec.applyPauliExp!(CuStateVec(h_sv), π/2, [PauliZ()], Int32[2], Int32[1], Int32[1])
        @test collect(sv.data) ≈ h_sv_result
        sv = @test_deprecated cuStateVec.applyPauliExp!(CuStateVec(h_sv), π/2, [PauliZ()], Int32[2], Int32[1])
        @test collect(sv.data) ≈ h_sv_result

        # |01⟩: ⟨Z₀⟩ = -1 and ⟨X₀⟩ = 0
        Z = elty[1 0; 0 -1]
        X = elty[0 1; 1 0]
        sv = CuStateVec(elty[0, 1, 0, 0])
        value, residual_norm = @test_deprecated cuStateVec.expectation(sv, Z, Int32[0])
        @test value isa Ref{Float64}
        @test value[] ≈ -1 atol=1e-6
        @test value[] == computeExpectation(sv, Z, Int32[0])
        @test residual_norm[] == 0

        exp_vals = @test_deprecated cuStateVec.expectationBatched(sv, 1, elty[vec(Z); vec(X)], 2, Int32[0])
        @test real(exp_vals) ≈ [-1; 0;;] atol=1e-6

        exp_vals = @test_deprecated cuStateVec.expectationsOnPauliBasis(sv, [Pauli[PauliZ()], Pauli[PauliX()]], [[0], [1]])
        @test exp_vals ≈ [-1, 0] atol=1e-6
    end

    full = ComplexF64.(1:32)
    sub_svs = [CuStateVec(full[8i+1:8i+8]) for i in 0:3]
    @test_deprecated cuStateVec.swapIndexBitsMultiDevice!(sub_svs, fill(device(), 4), [0 => 3], Int[], Int[],
                                                          cuStateVec.CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH)
    synchronize()
    swapbits(i) = (i & ~0b1001) | ((i & 1) << 3) | ((i >> 3) & 1)
    @test reduce(vcat, collect.(getfield.(sub_svs, :data))) == [full[swapbits(i) + 1] for i in 0:31]
end
