using BFloat16s: BFloat16

@testset "BFloat16" begin

@testset "load/store" begin
    function kernel(dst, src)
        i = threadIdx().x
        @inbounds dst[i] = src[i]
        return
    end
    src = CuArray(BFloat16[1, 2, 3, 4])
    dst = CUDA.zeros(BFloat16, 4)
    @cuda threads=4 kernel(dst, src)
    @test Array(dst) == BFloat16[1, 2, 3, 4]
end

# from the original JuliaGPU/CUDA.jl#2441 reproducer
@testset "in-kernel arithmetic" begin
    function kernel(x)
        @inbounds x[threadIdx().x] += BFloat16(1)
        return
    end
    x = CUDA.zeros(BFloat16, 4)
    @cuda threads=4 kernel(x)
    @test Array(x) == BFloat16[1, 1, 1, 1]
end

@testset "broadcast" begin
    @test testf((x, y) -> x .+ y, BFloat16[1, 2, 3], BFloat16[4, 5, 6])
    @test testf((x, y) -> x .- y, BFloat16[1, 2, 3], BFloat16[4, 5, 6])
    @test testf((x, y) -> x .* y, BFloat16[1, 2, 3], BFloat16[4, 5, 6])
    @test testf((x, y) -> x ./ y, BFloat16[1, 2, 3], BFloat16[4, 5, 6])
    # JuliaGPU/CUDA.jl#2441 (second comment): these crashed with `Cannot select`
    @test testf(x -> x .* x, BFloat16[1, 2, 3])
    @test testf(x -> x .^ 2, BFloat16[1, 2, 3])
end

# from the second LLVM "Cannot select" reproducer in JuliaGPU/CUDA.jl#2441
@testset "mixed-type promotion" begin
    function kernel(C, a, b)
        @inbounds C[] = a * b
        return
    end

    C32 = CUDA.zeros(Float32, 1)
    @cuda kernel(C32, BFloat16(2), Int32(3))
    @test Array(C32) == [6f0]

    @cuda kernel(C32, BFloat16(2), 3f0)
    @test Array(C32) == [6f0]

    # `cvt.f64.bf16` requires sm_90+
    if capability(device()) >= v"9.0"
        C64 = CUDA.zeros(Float64, 1)
        @cuda kernel(C64, BFloat16(2), 3.0)
        @test Array(C64) == [6.0]
    end
end

@testset "comparison" begin
    @test testf((x, y) -> x .== y, BFloat16[1, 2, 3], BFloat16[1, 5, 3])
    @test testf((x, y) -> x .<  y, BFloat16[1, 2, 3], BFloat16[2, 1, 3])
    @test testf((x, y) -> x .<= y, BFloat16[1, 2, 3], BFloat16[2, 2, 2])
end

@testset "conversion" begin
    @test testf(x -> Float32.(x), BFloat16[1, 2, 3])
    @test testf(x -> BFloat16.(x), Float32[1, 2, 3])
    @test testf(x -> BFloat16.(x), Int32[1, 2, 3])
end

@testset "reductions" begin
    @test testf(sum, BFloat16[1, 2, 3, 4])
    @test testf(prod, BFloat16[1, 2, 3, 4])
    # BFloat16 has only 7 bits of mantissa, so summing many values is lossy;
    # keep the input small enough that the result is exactly representable.
    @test testf(x -> sum(abs2, x), BFloat16[1, 2, 3, 4])
end

end  # @testset "BFloat16"
