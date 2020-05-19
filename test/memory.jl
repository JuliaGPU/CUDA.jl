CUDA.reclaim(1024)
CUDA.reclaim()

@test CUDA.@retry_reclaim(42, 42) == 42
@test CUDA.@retry_reclaim(42, 41) == 41

@test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)
