CUDA.alloc(0)

memcheck ||  @test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)

@testset "@allocated" begin
    @test (CUDA.@allocated CuArray{Int32}(undef,1)) == 4
end

@testset "@timed" begin
    out = CUDA.@timed CuArray{Int32}(undef, 1)
    @test isa(out.value, CuArray{Int32})
    @test out.gpu_bytes > 0
end

@testset "@time" begin
    ret, out = @grab_output CUDA.@time CuArray{Int32}(undef, 1)
    @test isa(ret, CuArray{Int32})
    @test occursin("1 GPU allocation: 4 bytes", out)

    ret, out = @grab_output CUDA.@time Base.unsafe_wrap(CuArray, CuPtr{Int32}(12345678), (2, 3))
    @test isa(ret, CuArray{Int32})
    @test !occursin("GPU allocation", out)
end

@testset "reclaim" begin
    CUDA.reclaim(1024)
    CUDA.reclaim()

    @test CUDA.@retry_reclaim(42, 42) == 42
    @test CUDA.@retry_reclaim(42, 41) == 41
end

@testset "timings" begin
    CUDA.enable_timings()
    CUDA.disable_timings()
end
