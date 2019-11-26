@testset "memory allocator" begin

CuArrays.reclaim(1024)
CuArrays.reclaim()

CuArrays.extalloc(()->())
CuArrays.extalloc(()->(); check=ex->true)
CuArrays.extalloc(()->(); nb=1)

@test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)
@test_throws OutOfGPUMemoryError CuArrays.extalloc() do
    CuArray{Int}(undef, 10^20)
end

end
