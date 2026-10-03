group = addgroup!(SUITE, "cuda")

let group = addgroup!(group, "synchronization")
    let group = addgroup!(group, "stream")
        group["blocking"] = @benchmarkable synchronize(blocking=true)
        group["auto"] = @benchmarkable synchronize()
        group["nonblocking"] = @benchmarkable synchronize(spin=false)
    end
    let group = addgroup!(group, "context")
        group["blocking"] = @benchmarkable device_synchronize(blocking=true)
        group["auto"] = @benchmarkable device_synchronize()
        group["nonblocking"] = @benchmarkable device_synchronize(spin=false)
    end
end

# many short operations, where launch overhead dominates
graph_array = CUDA.zeros(Float32, 1024)
graph_operations() = for _ in 1:10; graph_array .+= 1f0; end
graph_operations()
graph_exec = instantiate(capture(graph_operations))
let group = addgroup!(group, "graph")
    group["eager"] = @async_benchmarkable graph_operations()
    group["launch"] = @async_benchmarkable $graph_exec()
    group["capture"] = @benchmarkable capture(graph_operations)
end
