@testset "pointer" begin

# constructors

voidptr_a = CuPtr{Cvoid,AS.Host}(Int(0xDEADBEEF))
@test voidptr_a == CuHostPtr{Cvoid}(Int(0xDEADBEEF))

voidptr_b = CuPtr{Cvoid,AS.Host}(Int(0xCAFEBABE))
@test voidptr_b == CuHostPtr{Cvoid}(Int(0xCAFEBABE))

# getters
@test eltype(voidptr_a) == Cvoid

# comparisons
@test voidptr_a != voidptr_b


@testset "conversions" begin

# between regular and CUDA pointers

@test_throws ArgumentError convert(Ptr{Cvoid}, voidptr_device_a)

# between CUDA pointers

intptr_a = CuPtr{Int,AS.Host}(Int(0xDEADBEEF))
@test convert(typeof(intptr_a), voidptr_a) == intptr_a

end


end
