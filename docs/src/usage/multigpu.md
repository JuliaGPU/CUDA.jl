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
