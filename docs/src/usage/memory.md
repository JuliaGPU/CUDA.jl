# Memory Management

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

    The `cu` function is opinionated and converts input scalars to `Float32`. This is
    often a good call, as `Float64` and many other scalar types perform badly on the GPU.
    If this is unwanted, use `adapt` directly.


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
julia> CuArrays.memory_status()         # initial state
Effective GPU memory usage: 10.51% (1.654 GiB/15.744 GiB)
CuArrays GPU memory usage: 0 bytes
BinnedPool usage: 0 bytes (0 bytes allocated, 0 bytes cached)

julia> a = CuArray{Int}(undef, 1024);   # allocate 8KB

julia> CuArrays.memory_status()
Effective GPU memory usage: 10.52% (1.656 GiB/15.744 GiB)
CuArrays GPU memory usage: 8.000 KiB
BinnedPool usage: 8.000 KiB (8.000 KiB allocated, 0 bytes cached)

julia> a = nothing; GC.gc(true)

julia> CuArrays.memory_status()         # 8KB is now cached
Effective GPU memory usage: 10.52% (1.656 GiB/15.744 GiB)
CuArrays GPU memory usage: 8.000 KiB
BinnedPool usage: 8.000 KiB (0 bytes allocated, 8.000 KiB cached)
```

If for some reason you need all cached memory to be reclaimed, call `CuArrays.reclaim()`:

```julia-repl
julia> CuArrays.reclaim()
8192

julia> CuArrays.memory_status()
Effective GPU memory usage: 10.52% (1.656 GiB/15.744 GiB)
CuArrays GPU memory usage: 0 bytes
BinnedPool usage: 0 bytes (0 bytes allocated, 0 bytes cached)
```

!!! note

    It is should never be required to manually reclaim memory before performing any
    high-level GPU array operation: Functionality that allocates should itself call into the
    memory pool and free any cached memory if necessary. It is a bug if that operation
    runs into an out-of-memory situation only if not manually reclaiming memory beforehand.


### Environment variables

Several environment variables affect the behavior of the memory allocator:

- `CUARRAYS_MEMORY_POOL`: select a different memory pool. Several implementations are
  available:
  - `binned` (the default): cache memory in pow2-sized bins
  - `split`: caching pool that supports splitting allocations, designed to reduce pressure
    on the Julia garbage collector
  - `simple`: very simple caching layer for demonstration purposes
  - `none`: no pool at all, directly deferring to the CUDA allocator
- `CUARRAYS_MEMORY_LIMIT`: cap the amount of allocated GPU memory, in bytes.

These environment variables should be set before importing packages; changing them at run
time does not have any effect.
