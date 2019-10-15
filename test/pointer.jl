@testset "pointer" begin

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

end


@testset "GPU or CPU integration" begin

a = [1]
ccall(:clock, Nothing, (Ptr{Int},), a)
@test_throws Exception ccall(:clock, Nothing, (CuPtr{Int},), a)
ccall(:clock, Nothing, (CUDAdrv.PtrOrCuPtr{Int},), a)

buf = Mem.DeviceBuffer(NULL, 0, CuContext(C_NULL))
b = CuTestArray{eltype(a), ndims(a)}(buf, size(a))
ccall(:clock, Nothing, (CuPtr{Int},), b)
@test_throws Exception ccall(:clock, Nothing, (Ptr{Int},), b)
ccall(:clock, Nothing, (CUDAdrv.PtrOrCuPtr{Int},), b)

end


end
