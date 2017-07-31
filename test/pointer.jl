@testset "pointer" begin

const CU_NULL = CUDAdrv.OwnedPtr{Void}(C_NULL, CuContext(C_NULL))

# conversion to Ptr
@test_throws InexactError convert(Ptr{Void}, CU_NULL)
Base.unsafe_convert(Ptr{Void}, CU_NULL)

@test eltype(CUDAdrv.OwnedPtr{Void}) == Void
@test eltype(CU_NULL) == Void
@test isnull(CU_NULL)

@test_throws InexactError convert(Ptr{Void}, CU_NULL)
@test_throws InexactError convert(CUDAdrv.OwnedPtr{Void}, C_NULL)

null1 = CU_NULL + 1
@test null1 === CUDAdrv.OwnedPtr{Void}(C_NULL + 1, CU_NULL.ctx)
@test null1 - 1 === CU_NULL
@test null1 > CU_NULL

end
