@testset "pointer" begin

# conversion to Ptr
@test_throws InexactError convert(Ptr{Void}, CU_NULL)
Base.unsafe_convert(Ptr{Void}, CU_NULL)

let
    @test eltype(DevicePtr{Void}) == Void
    @test eltype(CU_NULL) == Void
    @test isnull(CU_NULL)

    @test_throws InexactError convert(Ptr{Void}, CU_NULL)
    @test_throws InexactError convert(DevicePtr{Void}, C_NULL)
end

end
