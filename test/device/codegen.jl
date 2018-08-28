@testset "code generation (relying on a device)" begin

############################################################################################

@testset "LLVM" begin

@testset "stripping invariant.load" begin
    @eval function device_codegen_invariant(ptr, x)
        i = CUDAnative.threadIdx_x()
        @inbounds unsafe_store!(ptr, x[i], 1)
    end

    buf = Mem.alloc(Float64)
    @cuda device_codegen_invariant(convert(Ptr{Float64}, buf.ptr), (1., 2., ))
    @test Mem.download(Float64, buf) == [1.]
end

@testset "stripping const TBAA" begin
    # this one is particularly nasty because it occurs in a nested function

    _a = rand(Int, 2, 1)
    b = ((1,9999),(1,9999))

    out_buf = Mem.alloc(Int, 2)
    a = Tuple(_a)

    @eval function device_codegen_nested_tbaa(out, a, b)
        i = threadIdx().x
        blockIdx().x
        @inbounds out[i,1] = a[i] + b[i][1]
    end

    @cuda threads=2 device_codegen_nested_tbaa(CuDeviceArray((2,1), CUDAnative.DevicePtr(convert(Ptr{Int}, out_buf.ptr))), a, b)
    @test Mem.download(Int, out_buf, 2) == (_a .+ 1)[1:2]
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
