using Base.CoreLogging: Debug

@testset "logging" begin
    cuDNN.enable_logging()
    try
        desc = Ref{cuDNN.cudnnTensorDescriptor_t}()

        # API calls are traced at debug level
        @test_logs (:debug, r"cudnnCreateTensorDescriptor") min_level=Debug match_mode=:any begin
            cuDNN.cudnnCreateTensorDescriptor(desc)
            CUDACore.flush_logs()
        end

        # failures are explained at error level
        logs, _ = Test.collect_test_logs(min_level=Debug) do
            cuDNN.unchecked_cudnnSetTensor4dDescriptor(desc[], cuDNN.CUDNN_TENSOR_NCHW,
                                                        cuDNN.CUDNN_DATA_FLOAT, -1, 1, 1, 1)
            CUDACore.flush_logs()
        end
        @test all(log -> log._module === cuDNN, logs)
        errors = filter(log -> log.level == Base.CoreLogging.Error, logs)
        @test !isempty(errors)
        @test occursin("CUDNN_STATUS_BAD_PARAM", first(errors).message)
        # lines of a message are joined
        @test occursin(r"cudnnSetTensor4dDescriptor.*\n.*Traceback"s, first(errors).message)

        cuDNN.enable_logging(false)
        @test_logs min_level=Debug begin
            cuDNN.cudnnDestroyTensorDescriptor(desc[])
            CUDACore.flush_logs()
        end
    finally
        cuDNN.enable_logging(false)
    end
end
