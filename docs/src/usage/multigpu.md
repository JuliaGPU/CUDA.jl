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
    as the first statement in your `@async` of `@spawn` block. This is due to the
    newly-created task deriving the active device from the previously-active task, and not
    from its parent task. This is expected to be fixed in the future.


### Memory management

When working with multiple devices, you need to be careful with allocated memory:
Allocations are tied to the device that was active when requesting the memory, and cannot be
used with another device. That means you cannot allocate a `CuArray`, switch devices, and
use that object. Similar restrictions apply to library objects, like CUFFT plans.

To avoid this difficulty, you can use unified memory that is accessible from all devices.
These APIs are available through high-level wrappers, but not exposed by the `CuArray`
constructors yet:

```julia
using CUDA

gpus = Int(length(devices()))

# generate CPU data
dims = (3,4,gpus)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

# CuArray doesn't support unified memory yet,
# so allocate our own buffers
buf_a = Mem.alloc(Mem.Unified, sizeof(a))
d_a = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_a), dims)
finalizer(d_a) do _
    Mem.free(buf_a)
end
copyto!(d_a, a)

buf_b = Mem.alloc(Mem.Unified, sizeof(b))
d_b = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_b), dims)
finalizer(d_b) do _
    Mem.free(buf_b)
end
copyto!(d_b, b)

buf_c = Mem.alloc(Mem.Unified, sizeof(a))
d_c = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_c), dims)
finalizer(d_c) do _
    Mem.free(buf_c)
end
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
