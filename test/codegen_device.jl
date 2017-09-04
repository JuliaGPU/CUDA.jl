@testset "code generation (relying on a device)" begin

############################################################################################

@testset "SASS" begin

@testset "basic reflection" begin
    @eval sass_valid_kernel() = nothing
    @eval sass_invalid_kernel() = 1

    @test code_sass(DevNull, sass_valid_kernel, Tuple{}) == nothing
    @test_throws ArgumentError code_sass(DevNull, sass_invalid_kernel, Tuple{}) == nothing
end

end

############################################################################################

end
