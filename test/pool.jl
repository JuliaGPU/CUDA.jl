@testcase "pool" begin

@testcase "essentials" begin
    CUDA.alloc(0)

    @not_if_memcheck @test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)
end

@testcase "@allocated" begin
    @test (CUDA.@allocated CuArray{Int32}(undef,1)) == 4
end

@testcase "@timed" begin
    out = CUDA.@timed CuArray{Int32}(undef, 1)
    @test isa(out.value, CuArray{Int32})
    # XXX: doesn't work with multithreading
    @test_skip out.gpu_bytes > 0
end

@testcase "@time" begin
    ret, out = @grab_output CUDA.@time CuArray{Int32}(undef, 1)
    @test isa(ret, CuArray{Int32})
    # XXX: doesn't work with multithreading
    @test_skip occursin("1 GPU allocation: 4 bytes", out)

    ret, out = @grab_output CUDA.@time Base.unsafe_wrap(CuArray, CuPtr{Int32}(12345678), (2, 3))
    @test isa(ret, CuArray{Int32})
    # XXX: doesn't work with multithreading
    @test_skip !occursin("GPU allocation", out)
end

@testcase "reclaim" begin
    CUDA.reclaim(1024)
    CUDA.reclaim()

    @test CUDA.@retry_reclaim(isequal(42), 42) == 42
    @test CUDA.@retry_reclaim(isequal(42), 41) == 41
end

@testcase "timings" begin
    CUDA.enable_timings()
    CUDA.disable_timings()
end

end
