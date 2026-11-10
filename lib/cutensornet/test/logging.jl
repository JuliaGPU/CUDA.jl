using Base.CoreLogging: Debug

@testset "logging" begin
    cuTensorNet.enable_logging()
    try
        # contract A[a,b] * B[b,c] -> C[a,c]
        n = 8
        A = CuArray(rand(Float32, n, n))
        B = CuArray(rand(Float32, n, n))
        C = CUDACore.zeros(Float32, n, n)
        net = cuTensorNet.CuNetwork{Float32}()
        cuTensorNet.append_tensor!(net, A, [1, 2])
        cuTensorNet.append_tensor!(net, B, [2, 3])
        cuTensorNet.set_output!(net, C, [1, 3])

        logs, _ = Test.collect_test_logs(min_level=Debug) do
            rehearse_contraction(net, 2^28)
            cuTensorNet.prepare_contraction!(net)
            cuTensorNet.contract!(net)
            synchronize()
            CUDACore.flush_logs()
        end
        @test all(log -> log._module === cuTensorNet, logs)
        @test any(log -> occursin("cutensornetNetworkContract", log.message), logs)

        cuTensorNet.enable_logging(false)
        @test_logs min_level=Debug begin
            cuTensorNet.contract!(net)
            synchronize()
            CUDACore.flush_logs()
        end
    finally
        cuTensorNet.enable_logging(false)
    end
end
