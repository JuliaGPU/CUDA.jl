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
