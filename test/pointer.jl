@testset "pointer" begin

# inner constructors

const generic_null = CUDAnative.DevicePtr{Void,AS.Generic}(C_NULL)
const global_null = CUDAnative.DevicePtr{Void,AS.Global}(C_NULL)
const local_null = CUDAnative.DevicePtr{Void,AS.Local}(C_NULL)

const C_NONNULL = Ptr{Void}(1)
const generic_nonnull = CUDAnative.DevicePtr{Void,AS.Generic}(C_NONNULL)
const global_nonnull = CUDAnative.DevicePtr{Void,AS.Global}(C_NONNULL)
const local_nonnull = CUDAnative.DevicePtr{Void,AS.Local}(C_NONNULL)

const C_ONE = Ptr{Int}(1)
const generic_one = CUDAnative.DevicePtr{Int,AS.Generic}(C_ONE)
const global_one = CUDAnative.DevicePtr{Int,AS.Global}(C_ONE)
const local_one = CUDAnative.DevicePtr{Int,AS.Local}(C_ONE)

# outer constructors
@test CUDAnative.DevicePtr{Void}(C_NULL) == generic_null
@test CUDAnative.DevicePtr(C_NULL) == generic_null

# getters
@test eltype(generic_null) == Void
@test addrspace(generic_null) == AS.Generic
@test isnull(generic_null)
@test !isnull(generic_nonnull)

# comparisons
@test generic_null != generic_one
@test generic_null != global_null
@test local_null != global_null


@testset "conversions" begin

# between regular and device pointers

@test_throws InexactError convert(Ptr{Void}, generic_null)
@test_throws InexactError convert(CUDAnative.DevicePtr{Void}, C_NULL)

@test Base.unsafe_convert(Ptr{Void}, generic_null) == C_NULL


# between device pointers

@test_throws InexactError convert(typeof(local_null), global_null) == local_null
@test convert(typeof(generic_null), generic_null) == generic_null
@test convert(typeof(global_null), global_null) == global_null
@test Base.unsafe_convert(typeof(local_null), global_null) == local_null

@test convert(typeof(global_null), global_one) == global_nonnull
@test convert(typeof(generic_null), global_one) == generic_nonnull
@test convert(typeof(global_null), generic_one) == global_nonnull
@test convert(CUDAnative.DevicePtr{Void}, global_one) == global_nonnull

end

end
