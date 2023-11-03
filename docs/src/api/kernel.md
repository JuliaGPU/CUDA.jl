# [Kernel programming](@id KernelAPI)

This section lists the package's public functionality that corresponds to special CUDA
functions for use in device code. It is loosely organized according to the [C language
extensions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions)
appendix from the CUDA C programming guide. For more information about certain intrinsics,
refer to the aforementioned NVIDIA documentation.


## Indexing and dimensions

```@docs
gridDim
blockIdx
blockDim
threadIdx
warpsize
laneid
active_mask
```


## Device arrays

CUDA.jl provides a primitive, lightweight array type to manage GPU data organized in an
plain, dense fashion. This is the device-counterpart to the `CuArray`, and implements (part
of) the array interface as well as other functionality for use _on_ the GPU:

```@docs
CuDeviceArray
CUDA.Const
```


## Memory types

### Shared memory

```@docs
CuStaticSharedArray
CuDynamicSharedArray
```

### Texture memory

```@docs
CuDeviceTexture
```


## Synchronization

```@docs
sync_threads
sync_threads_count
sync_threads_and
sync_threads_or
sync_warp
threadfence_block
threadfence
threadfence_system
```


## Time functions

```@docs
clock
nanosleep
```


## Warp-level functions

### Voting

The warp vote functions allow the threads of a given warp to perform a
reduction-and-broadcast operation. These functions take as input a boolean predicate from
each thread in the warp and evaluate it. The results of that evaluation are combined
(reduced) across the active threads of the warp in one different ways, broadcasting a single
return value to each participating thread.

```@docs
vote_all_sync
vote_any_sync
vote_uni_sync
vote_ballot_sync
```

### Shuffle

```@docs
shfl_sync
shfl_up_sync
shfl_down_sync
shfl_xor_sync
```


## Formatted Output

```@docs
@cushow
@cuprint
@cuprintln
@cuprintf
```


## Assertions

```@docs
@cuassert
```


## Atomics

A high-level macro is available to annotate expressions with:

```@docs
CUDA.@atomic
```

If your expression is not recognized, or you need more control, use the underlying
functions:

```@docs
CUDA.atomic_cas!
CUDA.atomic_xchg!
CUDA.atomic_add!
CUDA.atomic_sub!
CUDA.atomic_and!
CUDA.atomic_or!
CUDA.atomic_xor!
CUDA.atomic_min!
CUDA.atomic_max!
CUDA.atomic_inc!
CUDA.atomic_dec!
```


## Dynamic parallelism

Similarly to launching kernels from the host, you can use `@cuda` while passing
`dynamic=true` for launching kernels from the device. A lower-level API is available as
well:

```@docs
dynamic_cufunction
CUDA.DeviceKernel
```


## Cooperative groups

```@docs
CG
```


### Group construction and properties

```@docs
CG.thread_rank
CG.num_threads
CG.thread_block
```

```@docs
CG.this_thread_block
CG.group_index
CG.thread_index
CG.dim_threads
```

```@docs
CG.grid_group
CG.this_grid
CG.is_valid
CG.block_rank
CG.num_blocks
CG.dim_blocks
CG.block_index
```

```@docs
CG.coalesced_group
CG.coalesced_threads
CG.meta_group_rank
CG.meta_group_size
```

### Synchronization

```@docs
CG.sync
CG.barrier_arrive
CG.barrier_wait
```

## Data transfer

```@docs
CG.wait
CG.wait_prior
CG.memcpy_async
```


## Math

Many mathematical functions are provided by the `libdevice` library, and are wrapped by
CUDA.jl. These functions are used to implement well-known functions from the Julia standard
library and packages like SpecialFunctions.jl, e.g., calling the `cos` function will
automatically use `__nv_cos` from `libdevice` if possible.

Some functions do not have a counterpart in the Julia ecosystem, those have to be called
directly. For example, to call `__nv_logb` or `__nv_logbf` you use `CUDA.logb` in a kernel.

For a list of available functions, look at `src/device/intrinsics/math.jl`.


## WMMA

Warp matrix multiply-accumulate (WMMA) is a CUDA API to access Tensor Cores, a new hardware
feature in Volta GPUs to perform mixed precision matrix multiply-accumulate operations. The
interface is split in two levels, both available in the WMMA submodule: low level wrappers
around the LLVM intrinsics, and a higher-level API similar to that of CUDA C.

### LLVM Intrinsics

#### Load matrix
```@docs
WMMA.llvm_wmma_load
```

#### Perform multiply-accumulate
```@docs
WMMA.llvm_wmma_mma
```

#### Store matrix
```@docs
WMMA.llvm_wmma_store
```

### CUDA C-like API

#### Fragment

```@docs
WMMA.RowMajor
WMMA.ColMajor
WMMA.Unspecified
WMMA.FragmentLayout
WMMA.Fragment
```

#### WMMA configuration

```@docs
WMMA.Config
```

#### Load matrix

```@docs
WMMA.load_a
```

`WMMA.load_b` and `WMMA.load_c` have the same signature.

#### Perform multiply-accumulate

```@docs
WMMA.mma
```

#### Store matrix

```@docs
WMMA.store_d
```

#### Fill fragment

```@docs
WMMA.fill_c
```


## Other

```@docs
CUDA.align
```
