using CUDA: UnifiedMemory

# On devices without concurrent managed access (Windows, Jetson boards up to Orin), the CPU
# can only access unified memory while kernels are running if that memory is attached to
# the host. These tests access memory on the CPU while another stream keeps the GPU busy,
# which crashes when the memory is not; on other devices, they only check correctness.

work!(a) = (a .+= 1; nothing)

# compute-sanitizer serializes kernels, so the GPU wouldn't get to our own work
sanitize || @testset "CPU access while the GPU is busy" begin
    # compile beforehand
    a = CuArray{Int,1,UnifiedMemory}(undef, 128)
    fill!(a, 0)
    work!(a)
    synchronize()

    # memory that was used on the GPU, freed, and then allocated again
    a = CuArray{Int,1,UnifiedMemory}(undef, 128)
    fill!(unsafe_wrap(Array, a), 1)
    work!(a)
    CUDA.unsafe_free!(a)
    synchronize()
    while_gpu_busy() do
        b = CuArray{Int,1,UnifiedMemory}(undef, 128)
        fill!(unsafe_wrap(Array, b), 2)
        @test Array(b) == fill(2, 128)
    end

    # memory that was used while implicit synchronization was disabled
    a = CuArray{Int,1,UnifiedMemory}(undef, 128)
    fill!(unsafe_wrap(Array, a), 1)
    CUDA.enable_synchronization!(a, false)
    work!(a)
    synchronize()
    @test Array(a) == fill(2, 128)
    CUDA.enable_synchronization!(a)
    work!(a)
    while_gpu_busy() do
        @test Array(a) == fill(3, 128)
    end

    # memory last used by a task that has finished
    a = CuArray{Int,1,UnifiedMemory}(undef, 128)
    fill!(unsafe_wrap(Array, a), 1)
    wait(@async begin
        work!(a)
        synchronize()
    end)
    while_gpu_busy() do
        @test Array(a) == fill(2, 128)
    end
end

@testset "graph capture" begin
    # compile beforehand
    a = CuArray{Int,1,UnifiedMemory}(undef, 16)
    work!(a)
    synchronize()

    # memory first used on the GPU while capturing stays usable by every launch
    a = CuArray{Int,1,UnifiedMemory}(undef, 16)
    fill!(unsafe_wrap(Array, a), 10)
    graph = capture(() -> work!(a))
    exec = instantiate(graph)
    exec()
    @test Array(a) == fill(11, 16)
    exec()
    @test Array(a) == fill(12, 16)
    finalize(exec)
    finalize(graph)
    CUDA.reclaim()
    @test Array(a) == fill(12, 16)

    # once freed, its memory can be accessed on the CPU while the GPU is busy again
    CUDA.unsafe_free!(a)
    synchronize()
    sanitize || while_gpu_busy() do
        b = CuArray{Int,1,UnifiedMemory}(undef, 16)
        fill!(unsafe_wrap(Array, b), 1)
        @test Array(b) == fill(1, 16)
    end

    # memory allocated while capturing
    result = Ref{Any}()
    graph = capture() do
        result[] = CuArray{Int,1,UnifiedMemory}(undef, 16)
        fill!(result[], 7)
        work!(result[])
    end
    a = result[]
    exec = instantiate(graph)
    exec()
    @test Array(a) == fill(8, 16)
    exec()
    @test Array(a) == fill(8, 16)
    finalize(exec)
    finalize(graph)
    CUDA.reclaim()

    # accessing memory on the CPU while capturing, after the task that last used it on the
    # GPU has finished (whose stream may have been recycled into the capturing task)
    a = CuArray{Int,1,UnifiedMemory}(undef, 16)
    fill!(unsafe_wrap(Array, a), 20)
    wait(@async work!(a))
    graph = fetch(@async capture() do
        @test Array(a) == fill(21, 16)
    end)
    @test graph isa CuGraph
    finalize(graph)
end
