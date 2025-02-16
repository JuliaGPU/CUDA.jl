# Multiple GPUs

There are different ways of working with multiple GPUs: using one or more tasks, processes,
or systems. Although all of these are compatible with the Julia CUDA toolchain, the support
is a work in progress and the usability of some combinations can be significantly improved.


## Scenario 1: One GPU per process

The easiest solution that maps well onto Julia's existing facilities for distributed
programming, is to use one GPU per process

```julia
# spawn one worker per device
using Distributed, CUDA
addprocs(length(devices()))
@everywhere using CUDA

# assign devices
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end
```

Communication between nodes should happen via the CPU (the CUDA IPC APIs are available as
`CUDA.cuIpcOpenMemHandle` and friends, but not available through high-level wrappers).

Alternatively, one can use [MPI.jl](https://github.com/JuliaParallel/MPI.jl) together with
an CUDA-aware MPI implementation. In that case, `CuArray` objects can be passed as send and
receive buffers to point-to-point and collective operations to avoid going through the CPU.


## Scenario 2: Multiple GPUs per process

In a similar vein to the multi-process solution, one can work with multiple devices from
within a single process by calling `CUDA.device!` to switch to a specific device.
Furthermore, as the active device is a task-local property you can easily work with multiple
devices using one task per device. For more details, refer to the section on [Tasks and
threads](@ref).

!!! warning

    You currently need to re-set the device at the start of every task, i.e., call `device!`
    as one of the first statement in your `@async` or `@spawn` block:

    ```julia
    @sync begin
        @async begin
            device!(0)
            # do work on GPU 0 here
        end
        @async begin
            device!(1)
            # do work on GPU 1 here
        end
    end
    ```

    Without this, the newly-created task would use the same device as the
    previously-executing task, and not the parent task as could be expected. This is
    expected to be improved in the future using [context
    variables](https://github.com/JuliaLang/julia/pull/35833).


### Memory management

When working with multiple devices, you need to be careful with allocated memory:
Allocations are tied to the device that was active when requesting the memory, and cannot be
used with another device. That means you cannot allocate a `CuArray`, switch devices, and
use that object. Similar restrictions apply to library objects, like CUFFT plans.

To avoid this difficulty, you can use unified memory that is accessible from all devices:

```julia
using CUDA

gpus = Int(length(devices()))

# generate CPU data
dims = (3,4,gpus)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

# allocate and initialize GPU data
d_a = cu(a; unified=true)
d_b = cu(b; unified=true)
d_c = similar(d_a)
```

The data allocated here uses the GPU id as a the outermost dimension, which can be used to
extract views of contiguous memory that represent the slice to be processed by a single GPU:

```julia
for (gpu, dev) in enumerate(devices())
    device!(dev)
    @views d_c[:, :, gpu] .= d_a[:, :, gpu] .+ d_b[:, :, gpu]
end
```

Before downloading the data, make sure to synchronize the devices:

```julia
for dev in devices()
    # NOTE: normally you'd use events and wait for them
    device!(dev)
    synchronize()
end

using Test
c = Array(d_c)
@test a+b ≈ c
```

### Example: Large Matrix Vector Multiply 

A simple example that compares performance between a single GPU against
multiple-GPU environment for large matrix-vector multiplies.

```julia
using CUDA, BenchmarkTools
n = 10000
m = 300000
W = cu(rand(Float32, (n,m)))
x = cu(rand(Float32, m))
y = cu(zeros(Float32, n))
@benchmark CUDA.@sync y .= W*x
```

Benchmark Results running on a single GPU:
```
BenchmarkTools.Trial:
  memory estimate:  123.75 KiB
  allocs estimate:  7717
  --------------
  minimum time:     15.338 ms (0.00% GC)
  median time:      15.391 ms (0.00% GC)
  mean time:        15.393 ms (0.17% GC)
  maximum time:     15.798 ms (0.00% GC)
  --------------
  samples:          325
  evals/sample:     1
```

In a multiple GPU environment, the dimensions are split up such so that
matrix-vector multiplies can be done on different devices in parallel. Note that
Julia is column major order, so we explicitly split up the work along the
columns of the matrix.

```julia
# setup cpu elements
n_gpus = 3
n = 10000
m = 300000
dims = (n, div(m,n_gpus), n_gpus)
W = rand(Float32, dims)
x = rand(Float32, (div(m,n_gpus), n_gpus))
y = zeros(Float32, n, n_gpus)

# copy to gpu
buf_w = Mem.alloc(Mem.Unified, sizeof(W))
d_w = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_w), dims)
finalizer(d_w) do _
    Mem.free(buf_w)
end
copyto!(d_w, W)

buf_x = Mem.alloc(Mem.Unified, sizeof(x))
d_x = unsafe_wrap(CuArray{Float32,2}, convert(CuPtr{Float32}, buf_x), size(x))
finalizer(d_x) do _
    Mem.free(buf_x)
end
copyto!(d_x, x)

buf_y = Mem.alloc(Mem.Unified, sizeof(y))
d_y = unsafe_wrap(CuArray{Float32,2}, convert(CuPtr{Float32}, buf_y), size(y))
finalizer(d_y) do _
    Mem.free(buf_y)
end
copyto!(d_y, y)

@benchmark CUDA.@sync begin
    @sync begin
        for (gpu, dev) in enumerate(devices())
            @async begin
                device!(dev)
                @views d_y[:,gpu] .= d_w[:,:,gpu] * d_x[:,gpu]
            end
        end
    end
    device_synchronize()
end
```

Benchmark results: 
```
BenchmarkTools.Trial:
  memory estimate:  9.08 KiB
  allocs estimate:  285
  --------------
  minimum time:     5.331 ms (0.00% GC)
  median time:      8.133 ms (0.00% GC)
  mean time:        7.046 ms (19.94% GC)
  maximum time:     22.511 ms (75.23% GC)
  --------------
  samples:          709
  evals/sample:     1
```

As expected the speedup with splitting the work across 3 GPUs results in
roughly a 2-3x speedup.


### Example: Reduce over large vector

Here we try a reduction over a large vector by summing over all elements. On a
single GPU we might do: 
```julia
using CUDA, BenchmarkTools
m = 30000000
x = cu(rand(Float32, m))
result = 0
@benchmark CUDA.@sync result = reduce(+, x)
```

Benchmark results:
```
BenchmarkTools.Trial:
  memory estimate:  2.34 KiB
  allocs estimate:  104
  --------------
  minimum time:     239.676 μs (0.00% GC)
  median time:      246.791 μs (0.00% GC)
  mean time:        257.546 μs (0.86% GC)
  maximum time:     98.832 ms (22.52% GC)
  --------------
  samples:          10000
  evals/sample:     1
```

Distributing the work on multiple gpus, we might do: 
```julia
using CUDA, BenchmarkTools

# cpu data
n_gpus = 3
m = 30000000
x = rand(Float32, (div(m,n_gpus), n_gpus))

# copy to gpu
cux = Array{CuArray{Float32, 1}, 1}()
for (gpu, dev) in enumerate(devices())
    device!(dev)
    push!(cux, x[:,gpu])
end

results = Vector{Any}(undef, n_gpus)

# revert back to default
device!(first(devices()))

@benchmark CUDA.@sync begin
    @sync begin
        for (gpu, dev) in enumerate(devices())
            @async begin
                device!(dev)
                results[gpu] = reduce(+, cux[gpu])
            end
        end
    end
    device_synchronize()
    result = reduce(+, results)
end
```

Benchmark results:
```
BenchmarkTools.Trial:
  memory estimate:  11.75 KiB
  allocs estimate:  402
  --------------
  minimum time:     410.079 μs (0.00% GC)
  median time:      5.131 ms (0.00% GC)
  mean time:        5.317 ms (0.00% GC)
  maximum time:     30.191 ms (0.00% GC)
  --------------
  samples:          942
  evals/sample:     1
```

Strangely enough, the median/mean time for multiple-GPUs is significantly higher
than that of a single GPU, however the maximum time is about 1/3 the single GPU
scenario. Why the large variance? 
