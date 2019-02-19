using CuArrays

v = [1, 2, 3]
v2 = CuArray(v)

@testset "accumulate! tests"
    @test cumsum(v2) isa CuVector
    @test cumprod(v2) isa CuVector

    @test cumsum(v2) == [1, 3, 6]
    @test cumprod(v2) == [1, 2, 6]
end
