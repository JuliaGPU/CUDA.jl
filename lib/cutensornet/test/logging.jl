using Base.CoreLogging: Debug

@testset "logging" begin
    cuTensorNet.enable_logging()
    try
        # contract A[a,b] * B[b,c] -> C[a,c]
        n = 8
        A = CuArray(rand(Float32, n, n))
        B = CuArray(rand(Float32, n, n))
        ctn = CuTensorNetwork(Float32, [Int32['a', 'b'], Int32['b', 'c']], [[n, n], [n, n]],
                              [C_NULL, C_NULL], Int32[0, 0], Int32['a', 'c'], [n, n], C_NULL)
        ctn.input_arrs = [A, B]

        logs, _ = Test.collect_test_logs(min_level=Debug) do
            info = rehearse_contraction(ctn, 2^28)
            ctn.output_arr = CUDACore.zeros(Float32, n, n)
            perform_contraction!(ctn, info, NoAutoTune())
            synchronize()
            CUDACore.flush_logs()
        end
        @test all(log -> log._module === cuTensorNet, logs)
        @test any(log -> occursin("cutensornetContractSlices", log.message), logs)

        cuTensorNet.enable_logging(false)
        @test_logs min_level=Debug begin
            info = rehearse_contraction(ctn, 2^28)
            perform_contraction!(ctn, info, NoAutoTune())
            synchronize()
            CUDACore.flush_logs()
        end
    finally
        cuTensorNet.enable_logging(false)
    end
end
