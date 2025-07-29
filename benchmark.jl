using CUDA
using Test

a = CuArray([1, 2, 3, 4, 5, 6])

CUDA.fresize!(a, 2)
@test length(a) == 2
@test Array(a) == [1, 2]

CUDA.fresize!(a, 5)
@test length(a) == 5
# @test Array(a)[1:3] == [1, 2, 3]
Array(a)

# CUDA.fresize!(a, 2)
# @test length(a) == 2
# @test Array(a)[1:2] == [1, 2]

# # we should be able to resize an unsafe_wrapped array too, as it replaces the buffer
# b = unsafe_wrap(CuArray{Int}, pointer(a), 2)
# CUDA.fresize!(b, 3)
# @test length(b) == 3
# @test Array(b)[1:2] == [1, 2]

# b = CuArray{Int}(undef, 0)
# @test length(b) == 0
# CUDA.fresize!(b, 1)
# @test length(b) == 1