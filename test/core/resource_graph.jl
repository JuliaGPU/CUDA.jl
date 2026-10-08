@testset "array finalizers and graph leases" begin
    for graph_first in (false, true)
        a = CUDA.ones(Int, 4)
        output = similar(a)
        work = (a, output) -> (a .+= 1; copyto!(output, a); nothing)
        work(a, output) # compile before capture
        data = a.data
        graph = capture(() -> work(a, output))
        exec = instantiate(graph)
        finalize(graph_first ? graph : a)
        CUDA.reclaim()
        exec()
        @test Array(output) == fill(3, 4)
        finalize(graph_first ? a : graph)
        CUDA.reclaim()
        @test data.freed
        exec()
        @test Array(output) == fill(4, 4)
        finalize(exec)
        CUDA.reclaim()
        CUDA.unsafe_free!(output)
    end

    a = CUDA.ones(Int, 4)
    output = similar(a)
    work = (a, output) -> (a .+= 1; copyto!(output, a); nothing)
    work(a, output)
    data = a.data
    graph = capture() do
        work(a, output)
        finalize(a)
        GC.gc(true)
        CUDACore.drain_retired()
        @test !data.freed
    end
    CUDA.reclaim()
    @test data.freed
    exec = instantiate(graph)
    finalize(graph)
    CUDA.reclaim()
    exec()
    @test Array(output) == fill(3, 4)
    finalize(exec)
    CUDA.reclaim()
    CUDA.unsafe_free!(output)
end
