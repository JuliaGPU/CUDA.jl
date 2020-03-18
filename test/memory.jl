@testset "memory allocator" begin

CuArrays.reclaim(1024)
CuArrays.reclaim()

@test CuArrays.@retry_reclaim(42, 42) == 42
@test CuArrays.@retry_reclaim(42, 41) == 41

@test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)

end
