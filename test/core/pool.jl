CUDA.pool_alloc(0)

@test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)

try
    CuArray{Int}(undef, 10^20)
catch e
    @test startswith(sprint(showerror, e), "Out of GPU memory")
end


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

    x = CuArray{Int32}(undef, 6)
    ret, out = @grab_output CUDA.@time Base.unsafe_wrap(CuArray, pointer(x), (2, 3))
    @test isa(ret, CuArray{Int32})
    @test !occursin("GPU allocation", out)
end

@testset "reclaim" begin
    CUDA.reclaim(1024)
    CUDA.reclaim()

    @test CUDA.retry_reclaim(isequal(42)) do
            42
        end == 42
    @test CUDA.retry_reclaim(isequal(42)) do
            41
        end == 41
end

@testset "pool_status" begin
    CUDA.pool_status(devnull)
    CUDA.used_memory()
    CUDA.cached_memory()
end

@testset "parse_limit" begin
    @test CUDA.parse_limit("8000kB") == UInt(8000*1000)
    @test CUDA.parse_limit("8000MB") == UInt(8000*1000*1000)
    @test CUDA.parse_limit("8GB")    == UInt(8*1000*1000*1000)

    @test CUDA.parse_limit("8KiB")   == UInt(8*1024)
    @test CUDA.parse_limit("8MiB")   == UInt(8*1024*1024)
    @test CUDA.parse_limit("8GiB")   == UInt(8*1024*1024*1024)
end
