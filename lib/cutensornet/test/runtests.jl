using Test, CUDA, CUTENSORNET
import CUTENSORNET: CuTensorNetwork, rehearse_contraction, perform_contraction!, AutoTune, NoAutoTune

@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

@info "CUTENSORNET version: $(CUTENSORNET.version()) (built for CUDA $(CUTENSORNET.cuda_version()))"

@testset "CUTENSORNET" begin
    n = 8
    m = 16
    k = 32
    #@testset for elty in [Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64, Int8, Int32, UInt8, UInt32]
    @testset for elty in [Float16, Float32]
        # make the simplest TN: two matrices
        A = rand(elty, n, k)
        B = rand(elty, k, m)
        A_modes = Int32[1, 2]
        B_modes = Int32[2, 3]
        A_extents = collect(size(A))
        B_extents = collect(size(B))
        A_strides = collect(strides(A))
        B_strides = collect(strides(B))
        A_aligns  = Int32(256)
        B_aligns  = Int32(256)
        
        C = zeros(elty, n, m)
        C_modes = Int32[1, 3]
        C_extents = collect(size(C))
        C_strides = collect(strides(C))
        C_aligns  = Int32(256)
        @testset for max_ws_size in [2^10, 2^20] # test that slicing works
            @testset for tuning in [AutoTune(), NoAutoTune()]
                ctn = CuTensorNetwork(elty, [A_modes, B_modes], [A_extents, B_extents], [A_strides, B_strides], [A_aligns, B_aligns], C_modes, C_extents, C_strides, C_aligns)
                info, plan = rehearse_contraction(ctn, max_ws_size)
                ctn.input_arrs = CuArray.([A, B])
                ctn.output_arr = CuArray(C)
                ctn = perform_contraction!(ctn, info, plan, max_ws_size, tuning)
                dC = ctn.output_arr
                @test collect(dC) ≈ A*B 
            end
        end
    end
end
