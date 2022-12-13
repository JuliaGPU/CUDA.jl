using Test

# work around JuliaLang/Pkg.jl#2500
if VERSION < v"1.8"
    test_project = first(Base.load_path())
    preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
    test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
    if isfile(preferences_file) && !isfile(test_preferences_file)
        cp(preferences_file, test_preferences_file)
    end
end

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using CUTENSORNET
@test CUTENSORNET.has_cutensornet()
@info "CUTENSORNET version: $(CUTENSORNET.version()) (built for CUDA $(CUTENSORNET.cuda_version()))"

import CUTENSORNET: CuTensorNetwork, rehearse_contraction, perform_contraction!, AutoTune, NoAutoTune

using TensorOperations

@testset "CUTENSORNET" begin
    n = 8
    m = 16
    k = 32
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "Simple serial" begin
            modesA = ['m', 'h', 'k', 'n']
            modesB = ['u', 'k', 'h']
            modesC = ['x', 'u', 'y']
            modesD = ['m', 'x', 'n', 'y']
            extent = Dict{Char, Int}()
            extent['m'] = 96;
            extent['n'] = 96;
            extent['u'] = 96;
            extent['h'] = 64;
            extent['k'] = 64;
            extent['x'] = 64;
            extent['y'] = 64;
            extentsA = [extent[mode] for mode in modesA]
            extentsB = [extent[mode] for mode in modesB]
            extentsC = [extent[mode] for mode in modesC]
            extentsD = [extent[mode] for mode in modesD]
            A = CUDA.rand(elty, extentsA...)
            B = CUDA.rand(elty, extentsB...)
            C = CUDA.rand(elty, extentsC...)
            raw_data_in = [A, B, C]
            modes_in = [Int32.(modesA), Int32.(modesB), Int32.(modesC)]
            extents_in = [extentsA, extentsB, extentsC]
            aligns_in = UInt32.([256, 256, 256])
            aligns_out = UInt32(256)
            ctn = CuTensorNetwork(elty, modes_in, extents_in, [C_NULL, C_NULL, C_NULL], Int32[0, 0, 0], Int32.(modesD), extentsD, C_NULL)
            @testset for max_ws_size in [2^28, 2^32]
                @testset for tuning in [NoAutoTune(), AutoTune()]
                    ctn.input_arrs = raw_data_in
                    info = rehearse_contraction(ctn, max_ws_size)
                    ctn.output_arr = CUDA.zeros(elty, extentsD...)
                    ctn = perform_contraction!(ctn, info, tuning)
                    @test size(ctn.output_arr) == tuple(extentsD...)
                    hA = collect(A)
                    hB = collect(B)
                    hC = collect(C)
                    hD = zeros(elty, extentsD...)
                    @tensor begin
                        hD[m, x, n, y] := hA[m, h, k, n] * hB[u, k, h] * hC[x, u, y]
                    end
                    D = collect(ctn.output_arr)
                    @test D ≈ hD
                end
            end
        end
    end
end
