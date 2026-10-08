# [Kernel programming](@id KernelAPI)

```@meta
CurrentModule = CUDACore
```

This section lists the package's public functionality that corresponds to special CUDA
functions for use in device code. It is loosely organized according to the [C/C++ language
extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
appendix from the [CUDA programming
guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). For more information about
certain intrinsics, refer to the aforementioned NVIDIA documentation.


## Indexing and dimensions

!!! note "Differences with corresponding C/C++ indexing variables"
    The indexing functions [`blockIdx`](@ref) and [`threadIdx`](@ref) have different
    starting index from the corresponding variables in the C/C++ extensions.
    Be careful when literally porting code from C/C++.

```@docs
gridDim
clusterIdx
clusterDim
blockIdxInCluster
linearBlockIdxInCluster
blockIdx
blockDim
threadIdx
warpsize
laneid
lanemask
active_mask
FULL_MASK
```

## Device arrays

CUDA.jl provides a primitive, lightweight array type to manage GPU data organized in an
plain, dense fashion. This is the device-counterpart to the `CuArray`, and implements (part
of) the array interface as well as other functionality for use _on_ the GPU:

```@docs
CuDeviceArray
Const
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
trigger_programmatic_launch_completion
grid_dependency_synchronize
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
CUDACore.@atomic
```

If your expression is not recognized, or you need more control, use the underlying
functions.

### Memory scopes

Each low-level function takes an optional trailing `scope` argument selecting the set of
threads the operation is atomic with respect to, mirroring CUDA C's scoped variants:

| CUDA.jl                             | CUDA C              | Atomic with respect to          |
|:------------------------------------|:--------------------|:--------------------------------|
| `atomic_add!(ptr, val, Val(:block))`  | `atomicAdd_block`   | threads in the same block       |
| `atomic_add!(ptr, val)`               | `atomicAdd`         | all threads on the device       |
| `atomic_add!(ptr, val, Val(:system))` | `atomicAdd_system`  | the device, other devices and the host |

Device scope is the default, and what `CUDA.@atomic` uses. System scope is needed only when
the CPU or another GPU concurrently accesses the same memory, requires compute capability
6.0 (7.2 on Tegra), and is not available on Pascal GPUs under Windows. The low-level
functions reject system scope at compile time on targets below 6.0 and on Pascal under
Windows. Memory allocation and platform support must also permit system-wide atomicity.

### Memory ordering

Like CUDA C's atomic functions, the low-level functions and `CUDA.@atomic` are relaxed: they
are atomic, but don't order the memory accesses around them. To synchronize threads, use a
fence (`threadfence_block`, `threadfence` or `threadfence_system`), or an ordered atomic
from [UnsafeAtomics.jl](https://github.com/JuliaConcurrent/UnsafeAtomics.jl), which CUDA.jl
uses to implement its atomics. UnsafeAtomics also provides atomic loads and stores, and
operations CUDA.jl doesn't (e.g. `nand`, or floating-point `fmin`/`fmax`):

```julia
using UnsafeAtomics

function kernel(data, flag)
    if blockIdx().x == 1
        unsafe_store!(pointer(data), 42)
        # publish `data`
        UnsafeAtomics.store!(pointer(flag), Int32(1), UnsafeAtomics.release, UnsafeAtomics.device)
    else
        # wait for `data`
        while UnsafeAtomics.load(pointer(flag), UnsafeAtomics.acquire, UnsafeAtomics.device) == 0
        end
        @cuprintln(unsafe_load(pointer(data)))
    end
    return
end
```

UnsafeAtomics defaults to the system scope; pass `UnsafeAtomics.workgroup` for CUDA's
block scope, `UnsafeAtomics.device`, or `UnsafeAtomics.system`. Before compute capability
7.0, which has no ordered atomic instructions, the back-end implements ordered atomics with
fences. That requires NVPTX_LLVM_Backend_jll 23.1.2+1 or later (update your environment if
needed): earlier builds reject ordered loads and stores there, and relax ordered
read-modify-write operations.

For atomic operations on array elements, [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl)'s
`@atomic`, which KernelAbstractions.jl uses, is preferred over `CUDA.@atomic`. It supports
orderings (sequentially consistent by default) and uses the device scope.

```@docs
CUDACore.atomic_cas!
CUDACore.atomic_xchg!
CUDACore.atomic_add!
CUDACore.atomic_sub!
CUDACore.atomic_and!
CUDACore.atomic_or!
CUDACore.atomic_xor!
CUDACore.atomic_min!
CUDACore.atomic_max!
CUDACore.atomic_inc!
CUDACore.atomic_dec!
```


## Dynamic parallelism

Similarly to launching kernels from the host, you can use `@cuda` while passing
`dynamic=true` for launching kernels from the device. A lower-level API is available as
well:

```@docs
dynamic_cufunction
DeviceKernel
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
CUDACore.align
```
