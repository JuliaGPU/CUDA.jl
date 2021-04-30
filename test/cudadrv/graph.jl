@testset "graph" begin

let A = CUDA.zeros(Int, 1)
    # ensure compilation
    A .+= 1
    @test Array(A) == [1]

    graph = capture() do
        A .+= 1
    end
    @test Array(A) == [1]

    exec = instantiate(graph)
    CUDA.launch(exec)
    @test Array(A) == [2]

    graph′ = capture() do
        A .+= 2
    end

    update(exec, graph′)
    CUDA.launch(exec)
    @test Array(A) == [4]
end

let A = CUDA.zeros(Int, 1)
    function iteration(A, val)
        # custom kernel to force compilation on the first iteration
        function kernel(a, val)
            a[] += val
            return
        end
        @cuda kernel(A, val)
        return
    end

    for i in 1:2
        @captured iteration(A, i)
    end
    @test Array(A) == [3]
end

end
