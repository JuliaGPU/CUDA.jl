# Kernel programming

When arrays operations are not flexible enough, you can write your own GPU kernels in Julia.
CUDA.jl aims to expose the full power of the CUDA programming model, i.e., at the same
level of abstraction as CUDA C/C++, albeit with some Julia-specific improvements.

As a result, writing kernels in Julia is very similar to writing kernels in CUDA C/C++. It
should be possible to learn CUDA programming from existing CUDA C/C++ resources, and apply
that knowledge to programming in Julia using CUDA.jl. Nontheless, this section will give a
brief overview of the most important concepts and their syntax.


## Defining and launching kernels

Kernels are written as ordinary Julia functions, returning `nothing`:

```julia
function my_kernel()
    return
end
```

To launch this kernel, use the `@cuda` macro:

```julia-repl
julia> @cuda my_kernel()
```

This automatically (re)compiles the `my_kernel` function and launches it on the current
GPU (selected by calling `device!`).

By passing the `launch=false` keyword argument to `@cuda`, it is possible to obtain a
callable object representing the compiled kernel. This can be useful for reflection and
introspection purposes:

```julia-repl
julia> k = @cuda launch=false my_kernel()
CUDA.HostKernel for my_kernel()

julia> CUDA.registers(k)
4

julia> k()
```


## Kernel inputs and outputs

GPU kernels cannot return values, and should always `return` or `return nothing` on all
code paths. To communicate values from a kernel, you can use a `CuArray`:

```julia
function my_kernel(a)
    a[1] = 42
    return
end
```

```julia-repl
julia> a = CuArray{Int}(undef, 1);

julia> @cuda my_kernel(a);

julia> a
1-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 42
```


## Launch configuration and indexing

Simply using `@cuda` only launches a single thread, which is not very useful. To launch more
threads, use the `threads` and `blocks` keyword arguments to `@cuda`, while using indexing
intrinsics in the kernel to differentiate the computation for each thread:

```julia-repl
julia> function my_kernel(a)
           i = threadIdx().x
           a[i] = 42
           return
       end

julia> a = CuArray{Int}(undef, 5);

julia> @cuda threads=length(a) my_kernel(a);

julia> a
5-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 42
 42
 42
 42
 42
```

As shown above, the `threadIdx` etc. values from CUDA C are available as functions returning
a `NamedTuple` with `x`, `y`, and `z` fields. The intrinsics return 1-based indices.


## Synchronization

To synchronize threads in a block, use the `sync_threads()` function. More advanced variants
that take a predicate are also available:

- `sync_threads_count(pred)`: returns the number of threads for which `pred` was true
- `sync_threads_and(pred)`: returns `true` if `pred` was true for all threads
- `sync_threads_or(pred)`: returns `true` if `pred` was true for any thread

To maintain multiple thread synchronization barriers, use the `barrier_sync` function,
which takes an integer argument to identify the barrier.

To synchronize lanes in a warp, use the `sync_warp()` function. This function takes a mask
to select which lanes to participate (this defaults to `FULL_MASK`).

If only a memory barrier is required, and not an execution barrier, use fence intrinsics:

- `threadfence_block`: ensure memory ordering for all threads in the block
- `threadfence`: the same, but for all threads on the device
- `threadfence_system`: the same, but including host threads and threads on peer devices


## Device arrays

Although the `CuArray` type is the main array type used in CUDA.jl to represent GPU arrays
and invoke operations on the device, it is a type that's only meant to be used from the
host. For example, certain operations will call into the CUBLAS library, which is a library
whose entrypoints are meant to be invoked from the CPU.

When passing a `CuArray` to a kernel, it will be converted to a `CuDeviceArray` object
instead, representing the same memory but implemented with GPU-compatible operations. The
API surface of this type is very limited, i.e., it only supports indexing and assignment,
and some basic operations like `view`, `reinterpret`, `reshape`, etc. Implementing higher
level operations like `map` would be a performance trap, as they would not make use of the
GPU's parallelism, but execute slowly on a single GPU thread.

### Shared memory

To communicate between threads, device arrays that are backed by shared memory can be
allocated using the `CuStaticSharedArray` function:

```julia-repl
julia> function reverse_kernel(a::CuDeviceArray{T}) where T
           i = threadIdx().x
           b = CuStaticSharedArray(T, 2)
           b[2-i+1] = a[i]
           sync_threads()
           a[i] = b[i]
           return
       end

julia> a = cu([1,2])
2-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 1
 2

julia> @cuda threads=2 reverse_kernel(a)

julia> a
2-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 2
 1
```

When the amount of shared memory isn't known beforehand, and you don't want to recompile
the kernel for each size, you can use the `CuDynamicSharedArray` type instead. This requires
you to pass the size of the shared memory (in bytes) as an argument to the kernel:

```julia-repl
julia> function reverse_kernel(a::CuDeviceArray{T}) where T
           i = threadIdx().x
           b = CuDynamicSharedArray(T, length(a))
           b[length(a)-i+1] = a[i]
           sync_threads()
           a[i] = b[i]
           return
       end

julia> a = cu([1,2,3])
3-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 1
 2
 3

julia> @cuda threads=length(a) shmem=sizeof(a) reverse_kernel(a)

julia> a
3-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 3
 2
 1
```

When needing multiple arrays of dynamic shared memory, pass an `offset` parameter to the
subsequent `CuDynamicSharedArray` constructors indicating the offset in bytes from the
start of the shared memory. The `shmem` keyword to `@cuda` should be the total amount of
shared memory used by all arrays.

### Bounds checking

By default, indexing a `CuDeviceArray` will perform bounds checking, and throw an error
when the index is out of bounds. This can be a costly operation, so make sure to use
`@inbounds` when you know the index is in bounds.


## Standard output

CUDA.jl kernels do not yet integrate with Julia's standard input/output, but we provide
some basic functions to print to the standard output from a kernel:

- `@cuprintf`: print a formatted string to standard output
- `@cuprint` and `@cuprintln`: print a string and any values to standard output
- `@cushow`: print the name and value of an object

The `@cuprintf` macro does not support all formatting options; refer to the NVIDIA
documentation on `printf` for more details. It is often more convenient to use `@cuprintln`
and rely on CUDA.jl to convert any values to their appropriate string representation:

```julia-repl
julia> @cuda threads=2 (()->(@cuprintln("Hello, I'm thread $(threadIdx().x)!"); return))()
Hello, I'm thread 1!
Hello, I'm thread 2!
```

To simply show a value, which can be useful during debugging, use `@cushow`:

```julia-repl
julia> @cuda threads=2 (()->(@cushow threadIdx().x; return))()
(threadIdx()).x = 1
(threadIdx()).x = 2
```

Note that these aren't full-blown implementations, and only support a very limited number
of types. As such, they should only be used for debugging purposes.


## Random numbers

The `rand` and `randn` functions are available for use in kernels, and will return a
random number sampled from a special GPU-compatible random number generator:

```julia-repl
julia> @cuda (()->(@cushow rand(); return))()
rand() = 0.191897
```

Although the API is very similar to the random number generators used on the CPU, there
are a few differences and considerations that stem from the design of a parallel RNG:

- the default RNG uses global state; it is undefined behavior to use multiple instances
- kernels automatically seed the RNG with a unique seed passed from the host, ensuring
  that multiple invocations of the same kernel will produce different results
- manual seeding is possible by calling `Random.seed!`, however, the RNG uses warp-shared
  state, so at least one thread per warp should seed, and all seeds within a warp should be
  identical
- in the case that subsequent kernel invocations should continue the sequence of random
  numbers, not only the seed but also the counter value should be configured manually
  using `Random.seed!`; refer to CUDA.jl's host-side RNG for an example


## Atomics

CUDA.jl provides atomic operations at two levels of abstraction:

- low-level, `atomic_` functions mapping directly on hardware instructions
- high-level, `CUDA.@atomic` expressions for convenient element-wise operations

The former is the safest way to use atomic operations, as it is stable and will not change
behavior in the future. The interface is restrictive though, only supporting what the
hardware provides, and requiring matching input types. The `CUDA.@atomic` API is much more
user friendly, but will disappear at some point when it integrates with the `@atomic` macro
in Julia Base.

### Low-level

The low-level atomic in trinsics take pointer inputs, which can be obtained from calling
the `pointer` function on a `CuArray`:

```julia-repl
julia> function atomic_kernel(a)
           CUDA.atomic_add!(pointer(a), Int32(1))
           return
       end

julia> a = cu(Int32[1])
1-element CuArray{Int32, 1, CUDA.DeviceMemory}:
 1

julia> @cuda atomic_kernel(a)

julia> a
1-element CuArray{Int32, 1, CUDA.DeviceMemory}:
 2
```

Supported atomic operations are:
- typical binary operations: `add`, `sub`, `and`, `or`, `xor`, `min`, `max`, `xchg`
- NVIDIA-specific binary operations: `inc`, `dec`
- compare-and-swap: `cas`

Refer to the documentation of these intrinsics for more information on the type support,
and hardware requirements.

### High-level

For more convenient atomic operations on arrays, CUDA.jl provides the `CUDA.@atomic` macro
which can be used with expressions that assign array elements:

```julia-repl
julia> function atomic_kernel(a)
           CUDA.@atomic a[1] += 1
           return
       end

julia> a = cu(Int32[1])
1-element CuArray{Int32, 1, CUDA.DeviceMemory}:
 1

julia> @cuda atomic_kernel(a)

julia> a
1-element CuArray{Int32, 1, CUDA.DeviceMemory}:
 2
```

This macro is much more lenient, automatically converting inputs to the appropriate type,
and falling back to an atomic compare-and-swap loop for unsupported operations. It however
may disappear once CUDA.jl integrates with the `@atomic` macro in Julia Base.


## Warp intrinsics

Most of CUDA's warp intrinsics are available in CUDA.jl, under similar names. Their
behavior is mostly identical as well, with the exception that they are 1-indexed, and that
they support more types by automatically converting and splitting (to some extent) inputs:

- indexing: `laneid`, `lanemask`, `active_mask`, `warpsize`
- shuffle: `shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync`
- voting: `vote_all_sync`, `vote_any_sync`, `vote_unisync`, `vote_ballot_sync`

Many of these intrinsics require a `mask` argument, which is a bit mask indicating which
lanes should participate in the operation. To default to all lanes, use the `FULL_MASK`
constant.


## Dynamic parallelism

Where kernels are normally launched from the host, using dynamic parallelism it is also
possible to launch kernels from within a kernel. This is useful for recursive algorithms,
or for algorithms that otherwise need to dynamically spawn new work.

Device-side launches are also done using the `@cuda` macro, but require setting the
`dynamic` keyword argument to `true`:

```julia-repl
julia> function outer()
           @cuprint("Hello ")
           @cuda dynamic=true inner()
           return
       end

julia> function inner()
           @cuprintln("World!")
           return
       end

julia> @cuda outer()
Hello World!
```

Within a kernel, only a very limited subset of the CUDA API is available:
- synchronization: `device_synchronize`
- streams: `CuDeviceStream` constructor, `unsafe_destroy!` destuctor;
  these streams can be passed to `@cuda` using the `stream` keyword argument


## Cooperative groups

With cooperative groups, it is possible to write parallel kernels that are not tied to a
specific thread configuration, instead making it possible to more dynamically partition
threads and communicate between groups of threads. This functionality is relative new in
CUDA.jl, and does not yet support all aspects of the cooperative groups programming model.

Essentially, instead of manually computing a thread index and using that to differentiate
computation, kernel functionality now queries a group it is part of, and can query the size,
rank, etc of that group:

```julia-repl
julia> function reverse_kernel(d::CuDeviceArray{T}) where {T}
           block = CG.this_thread_block()

           n = length(d)
           t = CG.thread_rank(block)
           tr = n-t+1

           s = @inbounds CuDynamicSharedArray(T, n)
           @inbounds s[t] = d[t]
           CG.sync(block)
           @inbounds d[t] = s[tr]

           return
       end

julia> a = cu([1,2,3])
3-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 1
 2
 3

julia> @cuda threads=length(a) shmem=sizeof(a) reverse_kernel(a)

julia> a
3-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 3
 2
 1
```

The following implicit groups are supported:
- thread blocks: `CG.this_thread_block()`
- grid group: `CG.this_grid()`
- warps: `CG.coalesced_threads()`

Support is currently lacking for the cluster and multi-grid implicit groups, as well as all
explicit (tiled, partitioned) groups.

Thread blocks are supported by all devices, in all kernels. Grid groups (`CG.this_grid()`)
can be used to synchronize the entire grid, which is normally not possible, but requires
additional care: kernels need to be launched cooperatively, using `@cuda cooperative=true`,
which is only supported on devices with compute capability 6.0 or higher. Also, cooperative
kernels can only launch as many blocks as there are SMs on the device.

### Indexing

Every kind of thread group supports the following indexing operations:

- `thread_rank`: returns the rank of the current thread within the group
- `num_threads`: returns the number of threads in the group

In addition, some group kinds support additional indexing operations:
- thread blocks: `group_index`, `thread_index`, `dim_threads`
- grid group: `block_rank`, `num_blocks`, `dim_blocks`, `block_index`
- coalesced group: `meta_group_rank`, `meta_group_size`

Refer to the docstrings of these functions for more details.

### Synchronization

Group objects support the `CG.sync` operation to synchronize threads within a group.

In addition, thread and grid groups support more fine-grained synchronization using
barriers: `CG.barrier_arrive` and `CG.barrier_wait`: Calling `barrier_arrive` returns a
token that needs to be passed to `barrier_wait` to synchronize.

### Collective operations

Certain collective operations (i.e. operations that need to be performed by multiple
threads) provide a more convenient API when using cooperative groups. For example, shuffle
intrinsics normally require a thread mask, but this can be replaced by a group object:

```julia
function reverse_kernel(d)
    cta = CG.this_thread_block()
    I = CG.thread_rank(cta)

    warp = CG.coalesced_threads()
    i = CG.thread_rank(warp)
    j = CG.num_threads(warp) - i + 1

    d[I] = CG.shfl(warp, d[I], j)

    return
end
```

The following collective operations are supported:
- shuffle: `shfl`, `shfl_down`, `shfl_up`
- voting: `vote_any`, `vote_all`, `vote_ballot`

### Data transfer

With thread blocks and coalesced groups, the `CG.memcpy_async` function is available to
perform asynchronous memory copies. Currently, only copies from device to shared memory are
accelerated, and only on devices with compute capability 8.0 or higher. However, the
implementation degrades gracefully and will fall back to a synchronizing copy:

```julia-repl
julia> function memcpy_kernel(input::AbstractArray{T}, output::AbstractArray{T},
                              elements_per_copy) where {T}
           tb = CG.this_thread_block()

           local_smem = CuDynamicSharedArray(T, elements_per_copy)
           bytes_per_copy = sizeof(local_smem)

           i = 1
           while i <= length(input)
               # this copy can sometimes be accelerated
               CG.memcpy_async(tb, pointer(local_smem), pointer(input, i), bytes_per_copy)
               CG.wait(tb)

               # do something with the data here

               # this copy is always a simple element-wise operation
               CG.memcpy_async(tb, pointer(output, i), pointer(local_smem), bytes_per_copy)
               CG.wait(tb)

               i += elements_per_copy
           end
       end

julia> a = cu([1, 2, 3, 4]);
julia> b = similar(a);
julia> nb = 2;

julia> @cuda shmem=sizeof(eltype(a))*nb memcpy_kernel(a, b, nb)

julia> b
4-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 1
 2
 3
 4
```

The above example waits for the copy to complete before continuing, but it is also possible
to have multiple copies in flight using the `CG.wait_prior` function, which waits for all
but the last N copies to complete.


## Warp matrix multiply-accumulate

Warp matrix multiply-accumulate (WMMA) is a cooperative operation to perform mixed precision
matrix multiply-accumulate on the tensor core hardware of recent GPUs. The CUDA.jl
interface is split in two levels, both available in the WMMA submodule: low level wrappers
around the LLVM intrinsics, and a higher-level API similar to that of CUDA C.

### Terminology

The WMMA operations perform a matrix multiply-accumulate. More concretely, it calculates ``D
= A \cdot B + C``, where ``A`` is a ``M \times K`` matrix, ``B`` is a ``K \times N`` matrix,
and ``C`` and ``D`` are ``M \times N`` matrices.

However, not all values of ``M``, ``N`` and ``K`` are allowed. The tuple ``(M, N, K)`` is
often called the "shape" of the multiply accumulate operation.

The multiply-accumulate consists of the following steps:
- Load the matrices ``A``, ``B`` and ``C`` from memory to registers using a WMMA load
  operation.
- Perform the matrix multiply-accumulate of ``A``, ``B`` and ``C`` to obtain ``D`` using a
  WMMA MMA operation. ``D`` is stored in hardware registers after this step.
- Store the result ``D`` back to memory using a WMMA store operation.

Note that WMMA is a warp-wide operation, which means that all threads in a warp must
cooperate, and execute the WMMA operations in lockstep. Failure to do so will result in
undefined behaviour.

Each thread in a warp will hold a part of the matrix in its registers. In WMMA parlance,
this part is referred to as a "fragment". Note that the exact mapping between matrix
elements and fragment is unspecified, and subject to change in future versions.

Finally, it is important to note that the resultant ``D`` matrix can be used as a ``C``
matrix for a subsequent multiply-accumulate. This is useful if one needs to calculate a sum
of the form ``\sum_{i=0}^{n} A_i B_i``, where ``A_i`` and ``B_i`` are matrices of the
correct dimension.

### LLVM Intrinsics

The LLVM intrinsics are accessible by using the one-to-one Julia wrappers. The return type
of each wrapper is the Julia type that corresponds closest to the return type of the LLVM
intrinsic. For example, LLVM's `[8 x <2 x half>]` becomes `NTuple{8, NTuple{2,
VecElement{Float16}}}` in Julia. In essence, these wrappers return the SSA values returned
by the LLVM intrinsic. Currently, all intrinsics that are available in LLVM 6, PTX 6.0 and
SM 70 are implemented.

These LLVM intrinsics are then lowered to the correct PTX instructions by the LLVM NVPTX
backend. For more information about the PTX instructions, please refer to the [PTX
Instruction Set Architecture
Manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions).

The LLVM intrinsics are subdivided in three categories:

- load: `WMMA.llvm_wmma_load`
- multiply-accumulate: `WMMA.llvm_wmma_mma`
- store: `WMMA.llvm_wmma_store`

### CUDA C-like API

The main difference between the CUDA C-like API and the lower level wrappers, is that the
former enforces several constraints when working with WMMA. For example, it ensures that the
``A`` fragment argument to the MMA instruction was obtained by a `load_a` call, and not by a
`load_b` or `load_c`. Additionally, it makes sure that the data type and storage layout of
the load/store operations and the MMA operation match.

The CUDA C-like API heavily uses Julia's dispatch mechanism. As such, the method names are
much shorter than the LLVM intrinsic wrappers, as most information is baked into the type of
the arguments rather than the method name.

Note that, in CUDA C++, the fragment is responsible for both the storage of intermediate
results and the WMMA configuration. All CUDA C++ WMMA calls are function templates that take
the resultant fragment as a by-reference argument. As a result, the type of this argument
can be used during overload resolution to select the correct WMMA instruction to call.

In contrast, the API in Julia separates the WMMA storage ([`WMMA.Fragment`](@ref)) and
configuration ([`WMMA.Config`](@ref)). Instead of taking the resultant fragment by
reference, the Julia functions just return it. This makes the dataflow clearer, but it also
means that the type of that fragment cannot be used for selection of the correct WMMA
instruction. Thus, there is still a limited amount of information that cannot be inferred
from the argument types, but must nonetheless match for all WMMA operations, such as the
overall shape of the MMA. This is accomplished by a separate "WMMA configuration" (see
[`WMMA.Config`](@ref)) that you create once, and then give as an argument to all intrinsics.

- fragment: `WMMA.Fragment`
- configuration: `WMMA.Config`
- load: `WMMA.load_a`, `WMMA.load_b`, `WMMA.load_c`
- fill: `WMMA.fill_c`
- multiply-accumulate: `WMMA.mma`
- store: `WMMA.store_d`

#### Element access and broadcasting

Similar to the CUDA C++ WMMA API, [`WMMA.Fragment`](@ref)s have an `x` member that can be
used to access individual elements. Note that, in contrast to the values returned by the
LLVM intrinsics, the `x` member is flattened. For example, while the `Float16` variants of
the `load_a` instrinsics return `NTuple{8, NTuple{2, VecElement{Float16}}}`, the `x` member
has type `NTuple{16, Float16}`.

Typically, you will only need to access the `x` member to perform elementwise operations.
This can be more succinctly expressed using Julia's broadcast mechanism. For example, to
double each element in a fragment, you can simply use:

```julia
frag = 2.0f0 .* frag
```
