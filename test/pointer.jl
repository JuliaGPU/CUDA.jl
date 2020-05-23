# constructors
voidptr_a = CuPtr{Cvoid}(Int(0xDEADBEEF))
@test reinterpret(Ptr{Cvoid}, voidptr_a) == Ptr{Cvoid}(Int(0xDEADBEEF))

# getters
@test eltype(voidptr_a) == Cvoid

# comparisons
voidptr_b = CuPtr{Cvoid}(Int(0xCAFEBABE))
@test voidptr_a != voidptr_b


@testset "conversions" begin

# between regular and CUDA pointers
@test_throws ArgumentError convert(Ptr{Cvoid}, voidptr_a)

# between CUDA pointers
intptr_a = CuPtr{Int}(Int(0xDEADBEEF))
@test convert(typeof(intptr_a), voidptr_a) == intptr_a

# convert back and forth from UInt 
intptr_b = CuPtr{Int}(Int(0xDEADBEEF))
@test convert(UInt, intptr_b) == 0xDEADBEEF
@test convert(CuPtr{Int}, Int(0xDEADBEEF)) == intptr_b
@test Int(intptr_b) == Int(0xDEADBEEF)

# pointer arithmetic
intptr_c = CuPtr{Int}(Int(0xDEADBEEF))
intptr_d = 2 + intptr_c
@test isless(intptr_c, intptr_d)
@test intptr_d - intptr_c == 2
@test intptr_d - 2 == intptr_c
end


@testset "GPU or CPU integration" begin

a = [1]
ccall(:clock, Nothing, (Ptr{Int},), a)
@test_throws Exception ccall(:clock, Nothing, (CuPtr{Int},), a)
ccall(:clock, Nothing, (CUDA.PtrOrCuPtr{Int},), a)

ptr = convert(CuPtr{eltype(a)}, CU_NULL)
b = CuArray{eltype(a), ndims(a)}(ptr, size(a))
ccall(:clock, Nothing, (CuPtr{Int},), b)
@test_throws Exception ccall(:clock, Nothing, (Ptr{Int},), b)
ccall(:clock, Nothing, (CUDA.PtrOrCuPtr{Int},), b)

end
