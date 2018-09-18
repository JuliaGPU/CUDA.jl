@testset "code generation (relying on a device)" begin

############################################################################################

@testset "LLVM" begin

@testset "stripping invariant.load" begin
    @eval function device_codegen_invariant(ptr, x)
        i = CUDAnative.threadIdx_x()
        @inbounds unsafe_store!(ptr, x[i], 1)
        return
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
        return
    end

    @cuda threads=2 device_codegen_nested_tbaa(CuDeviceArray((2,1), CUDAnative.DevicePtr(convert(Ptr{Int}, out_buf.ptr))), a, b)
    @test Mem.download(Int, out_buf, 2) == (_a .+ 1)[1:2]
end


@testset "ptxas-compatible control flow" begin
    @eval @noinline function throw_some()
        throw(42)
        return
    end

    @eval @inbounds function device_codegen_cfg_gpu_kernel(input, output, n)
        i = threadIdx().x

        temp = @cuStaticSharedMem(Int, 1)
        if i == 1
            1 <= n || throw_some()
            temp[1] = input
        end
        sync_threads()

        1 <= n || throw_some()
        unsafe_store!(output, temp[1], i)

        return
    end

    function device_codegen_cfg_gpu(input)
        output = Mem.alloc(Int, 2)

        @cuda threads=2 device_codegen_cfg_gpu_kernel(input, convert(Ptr{eltype(input)}, output.ptr), 99)

        return Mem.download(Int, output, 2)
    end

    function device_codegen_cfg_cpu(input)
        output = zeros(eltype(input), 2)

        for j in 1:2
            @inbounds output[j] = input
        end

        return output
    end

    input = rand(1:100)
    @test device_codegen_cfg_cpu(input) == device_codegen_cfg_gpu(input)
end

end

############################################################################################

@testset "SASS" begin

@testset "basic reflection" begin
    @eval sass_valid_kernel() = return
    @eval sass_invalid_kernel() = 1

    @test CUDAnative.code_sass(devnull, sass_valid_kernel, Tuple{}) == nothing
    @test_throws CUDAnative.KernelError CUDAnative.code_sass(devnull, sass_invalid_kernel, Tuple{})
end

end

############################################################################################

end
