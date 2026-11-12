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

            # a mask bit string for each state vector, here selecting bit 2 = 1 and 0
            abs2sum = abs2SumArrayBatched(sv, n_svs, bit_ordering, [1, 0], [2])
            h = abs2.(h_sv)
            @test abs2sum ≈ [h[5] + h[6], h[7] + h[8], h[9] + h[10], h[11] + h[12]]
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

        # bit strings that cover only some of the index bits
        # cuStateVec rejects norms above one, so normalize each state vector
        h_sv = mapreduce(v -> v ./ sqrt(sum(abs2, v)), vcat, (rand(elty, 8), rand(elty, 8)))
        masks = [[i & 0b011 == 0b01 for i in 0:7], [i & 0b011 == 0b10 for i in 0:7]]
        norms = Float64[sum(abs2, h_sv[1:8][masks[1]]), sum(abs2, h_sv[9:16][masks[2]])]
        sv = collapseByBitStringBatched!(CuStateVec(h_sv), 2, [1, 2], [0, 1], norms)
        expected = vcat(h_sv[1:8] .* masks[1] ./ sqrt(norms[1]), h_sv[9:16] .* masks[2] ./ sqrt(norms[2]))
        @test collect(sv.data) ≈ expected
    end
end
