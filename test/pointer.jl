@testset "pointer" begin

# inner constructors

voidptr_a = CuPtr{Cvoid}(Int(0xDEADBEEF))
generic_voidptr_a = CUDAnative.DevicePtr{Cvoid,AS.Generic}(voidptr_a)
global_voidptr_a = CUDAnative.DevicePtr{Cvoid,AS.Global}(voidptr_a)
local_voidptr_a = CUDAnative.DevicePtr{Cvoid,AS.Local}(voidptr_a)

voidptr_b = CuPtr{Cvoid}(Int(0xCAFEBABE))
generic_voidptr_b = CUDAnative.DevicePtr{Cvoid,AS.Generic}(voidptr_b)
global_voidptr_b = CUDAnative.DevicePtr{Cvoid,AS.Global}(voidptr_b)
local_voidptr_b = CUDAnative.DevicePtr{Cvoid,AS.Local}(voidptr_b)

intptr_b = convert(CuPtr{Int}, voidptr_b)
generic_intptr_b = CUDAnative.DevicePtr{Int,AS.Generic}(intptr_b)
global_intptr_b = CUDAnative.DevicePtr{Int,AS.Global}(intptr_b)
local_intptr_b = CUDAnative.DevicePtr{Int,AS.Local}(intptr_b)

# outer constructors
@test CUDAnative.DevicePtr{Cvoid}(voidptr_a) == generic_voidptr_a
@test CUDAnative.DevicePtr(voidptr_a) == generic_voidptr_a

# getters
@test eltype(generic_voidptr_a) == Cvoid
@test eltype(global_intptr_b) == Int
@test addrspace(generic_voidptr_a) == AS.Generic
@test addrspace(global_voidptr_a) == AS.Global
@test addrspace(local_voidptr_a) == AS.Local

# comparisons
@test generic_voidptr_a != global_voidptr_a
@test generic_voidptr_a != generic_intptr_b


@testset "conversions" begin

# between host and device pointers

@test convert(CuPtr{Cvoid}, generic_voidptr_a) == voidptr_a
@test convert(CUDAnative.DevicePtr{Cvoid}, voidptr_a) == generic_voidptr_a
@test convert(CUDAnative.DevicePtr{Cvoid,AS.Global}, voidptr_a) == global_voidptr_a


# between device pointers

@test_throws ArgumentError convert(typeof(local_voidptr_a), global_voidptr_a)
@test convert(typeof(generic_voidptr_a), generic_voidptr_a) == generic_voidptr_a
@test convert(typeof(global_voidptr_a), global_voidptr_a) == global_voidptr_a
@test Base.unsafe_convert(typeof(local_voidptr_a), global_voidptr_a) == local_voidptr_a

@test convert(typeof(global_voidptr_a), global_intptr_b) == global_voidptr_b
@test convert(typeof(generic_voidptr_a), global_intptr_b) == generic_voidptr_b
@test convert(typeof(global_voidptr_a), generic_intptr_b) == global_voidptr_b

@test convert(CUDAnative.DevicePtr{Cvoid}, global_intptr_b) == global_voidptr_b

end

end
