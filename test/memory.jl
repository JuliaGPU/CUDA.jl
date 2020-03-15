@testset "memory allocator" begin

CuArrays.reclaim(1024)
CuArrays.reclaim()

@test CuArrays.@retry_reclaim(42, return 42) == 42
@test CuArrays.@retry_reclaim(42, return 41) == 41

@test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)
@test_throws OutOfGPUMemoryError CuArrays.extalloc() do
    CuArray{Int}(undef, 10^20)
end

end
