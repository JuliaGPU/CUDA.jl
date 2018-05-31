@testset "code generation (relying on a device)" begin

############################################################################################

@testset "SASS" begin

@testset "basic reflection" begin
    @eval sass_valid_kernel() = nothing
    @eval sass_invalid_kernel() = 1

    @test CUDAnative.code_sass(devnull, sass_valid_kernel, Tuple{}) == nothing
    @test_throws CUDAnative.CompilerError CUDAnative.code_sass(devnull, sass_invalid_kernel, Tuple{})
end

end

############################################################################################

end
