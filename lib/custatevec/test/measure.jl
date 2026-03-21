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
