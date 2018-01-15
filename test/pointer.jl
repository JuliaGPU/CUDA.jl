@testset "pointer" begin

# inner constructors

generic_null = CUDAnative.DevicePtr{Cvoid,AS.Generic}(C_NULL)
global_null = CUDAnative.DevicePtr{Cvoid,AS.Global}(C_NULL)
local_null = CUDAnative.DevicePtr{Cvoid,AS.Local}(C_NULL)

C_NONNULL = Ptr{Cvoid}(1)
generic_nonnull = CUDAnative.DevicePtr{Cvoid,AS.Generic}(C_NONNULL)
global_nonnull = CUDAnative.DevicePtr{Cvoid,AS.Global}(C_NONNULL)
local_nonnull = CUDAnative.DevicePtr{Cvoid,AS.Local}(C_NONNULL)

C_ONE = Ptr{Int}(1)
generic_one = CUDAnative.DevicePtr{Int,AS.Generic}(C_ONE)
global_one = CUDAnative.DevicePtr{Int,AS.Global}(C_ONE)
local_one = CUDAnative.DevicePtr{Int,AS.Local}(C_ONE)

# outer constructors
@test CUDAnative.DevicePtr{Cvoid}(C_NULL) == generic_null
@test CUDAnative.DevicePtr(C_NULL) == generic_null

# getters
@test eltype(generic_null) == Cvoid
@test addrspace(generic_null) == AS.Generic
@test isnull(generic_null)
@test !isnull(generic_nonnull)

# comparisons
@test generic_null != generic_one
@test generic_null != global_null
@test local_null != global_null


@testset "conversions" begin

# between regular and device pointers

@test_throws InexactError convert(Ptr{Cvoid}, generic_null)
@test_throws InexactError convert(CUDAnative.DevicePtr{Cvoid}, C_NULL)

@test Base.unsafe_convert(Ptr{Cvoid}, generic_null) == C_NULL


# between device pointers

@test_throws InexactError convert(typeof(local_null), global_null) == local_null
@test convert(typeof(generic_null), generic_null) == generic_null
@test convert(typeof(global_null), global_null) == global_null
@test Base.unsafe_convert(typeof(local_null), global_null) == local_null

@test convert(typeof(global_null), global_one) == global_nonnull
@test convert(typeof(generic_null), global_one) == generic_nonnull
@test convert(typeof(global_null), generic_one) == global_nonnull
@test convert(CUDAnative.DevicePtr{Cvoid}, global_one) == global_nonnull

end

end
