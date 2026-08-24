# the kernel state is part of the launch ABI and gets forwarded to dynamic parallelism
# child launches, so it must stay compact
@test sizeof(CUDACore.KernelState) == 2sizeof(UInt)

@testset "idle backoff" begin
    # This used to call `usleep` unconditionally, which is unavailable on Windows.
    @test_nowarn CUDACore.hostcall_backoff()
end
