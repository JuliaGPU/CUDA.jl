@testset "code generation (relying on a device)" begin

############################################################################################

@testset "LLVM" begin

@testset "invariant tuple passing" begin
    @eval function kernel(ptr, x)
        i = CUDAnative.threadIdx_x()
        @inbounds unsafe_store!(ptr, x[i], 1)
    end

    buf = Mem.alloc(Float64)
    @cuda kernel(convert(Ptr{Float64}, buf.ptr), (1., 2., ))
    @test Mem.download(Float64, buf) == [1.]
end

end

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
