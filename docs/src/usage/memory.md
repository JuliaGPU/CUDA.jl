# Memory management

A crucial aspect of working with a GPU is managing the data on it. The `CuArray` type is the
primary interface for doing so: Creating a `CuArray` will allocate data on the GPU, copying
elements to it will upload, and converting back to an `Array` will download values to the
CPU:

```julia
# generate some data on the CPU
cpu = rand(Float32, 1024)

# allocate on the GPU
gpu = CuArray{Float32}(undef, 1024)

# copy from the CPU to the GPU
copyto!(gpu, cpu)

# download and verify
@test cpu == Array(gpu)
```

A shorter way to accomplish these operations is to call the copy constructor, i.e.
`CuArray(cpu)`.


## Type-preserving upload

In many cases, you might not want to convert your input data to a dense `CuArray`. For
example, with array wrappers you will want to preserve that wrapper type on the GPU and only
upload the contained data. The [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) package does
exactly that, and contains a list of rules on how to unpack and reconstruct types like array
wrappers so that we can preserve the type when, e.g., uploading data to the GPU:

```julia-repl
julia> cpu = Diagonal([1,2])     # wrapped data on the CPU
2×2 Diagonal{Int64,Array{Int64,1}}:
 1  ⋅
 ⋅  2

julia> using Adapt

julia> gpu = adapt(CuArray, cpu) # upload to the GPU, keeping the wrapper intact
2×2 Diagonal{Int64,CuArray{Int64,1,Nothing}}:
 1  ⋅
 ⋅  2
```

Since this is a very common operation, the `cu` function conveniently does this for you:

```julia-repl
julia> cu(cpu)
2×2 Diagonal{Float32,CuArray{Float32,1,Nothing}}:
 1.0   ⋅
  ⋅   2.0
```

!!! warning

    The `cu` function is opinionated and converts input most floating-point scalars to
    `Float32`. This is often a good call, as `Float64` and many other scalar types perform
    badly on the GPU. If this is unwanted, use `adapt` directly.


## Unified memory

The `CuArray` constructor and the `cu` function default to allocating device memory, which
can be accessed only from the GPU. It is also possible to allocate unified memory, which is
accessible from both the CPU and GPU with the driver taking care of data movement:

```julia-repl
julia> cpu = [1,2]
2-element Vector{Int64}:
 1
 2

julia> gpu = CuVector{Int,CUDA.UnifiedMemory}(cpu)
2-element CuArray{Int64, 1, CUDA.UnifiedMemory}:
 1
 2

julia> gpu = cu(cpu; unified=true)
2-element CuArray{Int64, 1, CUDA.UnifiedMemory}:
 1
 2
```

Using unified memory has several advantages: it is possible to allocate more memory than the
GPU has available, and the memory can be accessed efficiently from the CPU, either directly
or by wrapping the `CuArray` using an `Array`:

```julia-repl
julia> gpu[1]  # no scalar indexing error!
1

julia> cpu_again = unsafe_wrap(Array, gpu)
2-element Vector{Int64}:
 1
 2
```

This may make it significantly easier to port code to the GPU, as you can incrementally
port parts of your application without having to worry about executing CPU code, or
triggering an `AbstractArray` fallback. It may come at a cost however, as unified memory
needs to be paged in and out of the GPU memory, and cannot be allocated asynchronously. To
alleviate this cost, CUDA.jl automatically prefetches unified memory when passing it to a
kernel.

On recent systems (CUDA 12.2 with the open-source NVIDIA driver) it is also possible to do
the reverse, and access CPU memory from the GPU without having to explicitly allocate
unified memory using the `CuArray` constructor or `cu` function:

```julia-repl
julia> cpu = [1,2];

julia> gpu = unsafe_wrap(CuArray, cpu)
2-element CuArray{Int64, 1, CUDA.UnifiedMemory}:
 1
 2

julia> gpu .+= 1;

julia> cpu
2-element Vector{Int64}:
 2
 3
```

Right now, CUDA.jl still defaults to allocating device memory, but this may change in the
future. If you want to change the default behavior, you can set the `default_memory`
preference to `unified` or `host` instead of `device`.


## Garbage collection

Instances of the `CuArray` type are managed by the Julia garbage collector. This means that
they will be collected once they are unreachable, and the memory hold by it will be
repurposed or freed. There is no need for manual memory management, just make sure your
objects are not reachable (i.e., there are no instances or references).

### Memory pool

Behind the scenes, a memory pool will hold on to your objects and cache the underlying
memory to speed up future allocations. As a result, your GPU might seem to be running out of
memory while it isn't. When memory pressure is high, the pool will automatically free cached
objects:

```julia-repl
julia> CUDA.pool_status()             # initial state
Effective GPU memory usage: 16.12% (2.537 GiB/15.744 GiB)
Memory pool usage: 0 bytes (0 bytes reserved)

julia> a = CuArray{Int}(undef, 1024);   # allocate 8KB

julia> CUDA.pool_status()
Effective GPU memory usage: 16.35% (2.575 GiB/15.744 GiB)
Memory pool usage: 8.000 KiB (32.000 MiB reserved)

julia> a = nothing; GC.gc(true)

julia> CUDA.pool_status()             # 8KB is now cached
Effective GPU memory usage: 16.34% (2.573 GiB/15.744 GiB)
Memory pool usage: 0 bytes (32.000 MiB reserved)

```

If for some reason you need all cached memory to be reclaimed, call `CUDA.reclaim()`:

```julia-repl
julia> CUDA.reclaim()

julia> CUDA.pool_status()
Effective GPU memory usage: 16.17% (2.546 GiB/15.744 GiB)
Memory pool usage: 0 bytes (0 bytes reserved)
```

!!! note

    It should never be required to manually reclaim memory before performing any
    high-level GPU array operation: Functionality that allocates should itself call into the
    memory pool and free any cached memory if necessary. It is a bug if that operation
    runs into an out-of-memory situation only if not manually reclaiming memory beforehand.


!!! note

    If you need to disable the memory pool, e.g. because of incompatibility with certain
    CUDA APIs, set the environment variable `JULIA_CUDA_MEMORY_POOL` to `none` before
    importing CUDA.jl.


### Memory limits

If you're sharing a GPU with other users or applications, you might want to limit how much
memory is used. By default, CUDA.jl will configure the memory pool to use all available
device memory. You can change this using two environment variables:

* `JULIA_CUDA_SOFT_MEMORY_LIMIT`: This is an advisory limit, used to configure the memory
  pool. If you set this to a nonzero value, the memory pool will attempt to release cached
  memory until memory use falls below this limit. Note that this only happens at specific
  synchronization points, so memory use may temporarily exceed this limit. In addition,
  this limit is incompatible with `JULIA_CUDA_MEMORY_POOL=none`.
* `JULIA_CUDA_HARD_MEMORY_LIMIT`: This is a hard limit, checked before every allocation.
  On older versions of CUDA, before v12.2, this is a relatively expensive limit, so it is
  recommended to first try to use the soft limit.

The value of these variables can be formatted as a numer of bytes, optionally followed by
a unit, or as a percentage of the total device memory. Examples: `100M`, `50%`, `1.5GiB`,
`10000`.


### Avoiding GC pressure

When your application performs a lot of memory operations, the time spent during GC might
increase significantly. This happens more often than it does on the CPU because GPUs tend to
have smaller memories and more frequently run out of it. When that happens, CUDA invokes
the Julia garbage collector, which then needs to scan objects to see if they can be freed to
get back some GPU memory.

To avoid having to depend on the Julia GC to free up memory, you can directly inform
CUDA.jl when an allocation can be freed (or reused) by calling the `unsafe_free!`
method. Once you've done so, you cannot use that array anymore:

```julia-repl
julia> a = CuArray([1])
1-element CuArray{Int64,1,Nothing}:
 1

julia> CUDA.unsafe_free!(a)

julia> a
1-element CuArray{Int64,1,Nothing}:
Error showing value of type CuArray{Int64,1,Nothing}:
ERROR: AssertionError: Use of freed memory
```


## Batching iterator

If you are dealing with data sets that are too large to fit on the GPU all at once, you can
use `CuIterator` to batch operations:

```julia
julia> batches = [([1], [2]), ([3], [4])]

julia> for (batch, (a,b)) in enumerate(CuIterator(batches))
         println("Batch $batch: ", a .+ b)
       end
Batch 1: [3]
Batch 2: [7]
```

For each batch, every argument (assumed to be an array-like) is uploaded to the GPU using
the `adapt` mechanism from above. Afterwards, the memory is eagerly put back in the CUDA
memory pool using `unsafe_free!` to lower GC pressure.
