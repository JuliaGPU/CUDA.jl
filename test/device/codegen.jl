@testset "code generation (relying on a device)" begin

############################################################################################

@testset "LLVM" begin

@testset "stripping invariant.load" begin
    function kernel(ptr, x)
        i = CUDAnative.threadIdx_x()
        @inbounds unsafe_store!(ptr, x[i], 1)
        return
    end

    arr = CuTestArray(zeros(Float64))
    ptr = convert(CuPtr{Float64}, arr.buf)

    @cuda kernel(ptr, (1., 2., ))
    @test Array(arr)[] == 1.
end

@testset "stripping const TBAA" begin
    # this one is particularly nasty because it occurs in a nested function

    _a = rand(Int, 2, 1)
    b = ((1,9999),(1,9999))

    out = CuTestArray(zeros(Int, 2,1))
    a = Tuple(_a)

    function kernel(out, a, b)
        i = threadIdx().x
        blockIdx().x
        @inbounds out[i,1] = a[i] + b[i][1]
        return
    end

    @cuda threads=2 kernel(out, a, b)
    @test Array(out) == (_a .+ 1)
end


@testset "ptxas-compatible control flow" begin
    @noinline function throw_some()
        throw(42)
        return
    end

    @inbounds function kernel(input, output, n)
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

    function gpu(input)
        output = CuTestArray(zeros(eltype(input), 2))
        ptr = convert(CuPtr{eltype(input)}, output.buf)
        ptr = reinterpret(Ptr{eltype(input)}, ptr)

        @cuda threads=2 kernel(input, ptr, 99)

        return Array(output)
    end

    function cpu(input)
        output = zeros(eltype(input), 2)

        for j in 1:2
            @inbounds output[j] = input
        end

        return output
    end

    input = rand(1:100)
    @test cpu(input) == gpu(input)
end

end

############################################################################################

@testset "SASS" begin

@testset "basic reflection" begin
    valid_kernel() = return
    invalid_kernel() = 1

    @test CUDAnative.code_sass(devnull, valid_kernel, Tuple{}) == nothing
    @test_throws CUDAnative.KernelError CUDAnative.code_sass(devnull, invalid_kernel, Tuple{})
end

end

############################################################################################

end
