# CUDA

This section lists the package's public functionality that corresponds to special CUDA
functions for use in device code. It is loosely organized according to the [C language
extensions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions)
appendix from the CUDA C programming guide. For more information about certain intrinsics,
refer to the aforementioned NVIDIA documentation.


## Indexing and Dimensions

```@docs
CUDAnative.gridDim
CUDAnative.blockIdx
CUDAnative.blockDim
CUDAnative.threadIdx
CUDAnative.warpsize
```


## Memory Types

### Shared Memory

```@docs
CUDAnative.@cuStaticSharedMem
CUDAnative.@cuDynamicSharedMem
```


## Synchronization

```@docs
CUDAnative.sync_threads
CUDAnative.sync_warp
CUDAnative.threadfence_block
CUDAnative.threadfence
CUDAnative.threadfence_system
```

## Clock & Sleep

```@docs
CUDAnative.clock
CUDAnative.nanosleep
```

## Warp Vote

The warp vote functions allow the threads of a given warp to perform a
reduction-and-broadcast operation. These functions take as input a boolean predicate from
each thread in the warp and evaluate it. The results of that evaluation are combined
(reduced) across the active threads of the warp in one different ways, broadcasting a single
return value to each participating thread.

```@docs
CUDAnative.vote_all
CUDAnative.vote_any
CUDAnative.vote_ballot
```


## Warp Shuffle

```@docs
CUDAnative.shfl
CUDAnative.shfl_up
CUDAnative.shfl_down
CUDAnative.shfl_xor
```

If using CUDA 9.0, and PTX ISA 6.0 is supported, synchronizing versions of these
intrinsics are available as well:

```@docs
CUDAnative.shfl_sync
CUDAnative.shfl_up_sync
CUDAnative.shfl_down_sync
CUDAnative.shfl_xor_sync
```


## Formatted Output

```@docs
CUDAnative.@cuprintf
```


## Assertions

```@docs
CUDAnative.@cuassert
```


## CUDA runtime

Certain parts of the CUDA API are available for use on the GPU, for example to launch
dynamic kernels or set-up cooperative groups. Coverage of this part of the API, provided by
the `libcudadevrt` library, is under development and contributions are welcome.

Calls to these functions are often ambiguous with their host-side equivalents as implemented
in CUDAdrv. To avoid confusion, you need to prefix device-side API interactions with the
CUDAnative module, e.g., `CUDAnative.synchronize`.

```@docs
CUDAnative.device_synchronize
```


## Math

Many mathematical functions are provided by the `libdevice` library, and are wrapped by
CUDAnative.jl. These functions implement interfaces that are similar to existing functions
in `Base`, albeit often with support for fewer types.

To avoid confusion with existing implementations in `Base`, you need to prefix calls to this
library with the CUDAnative module. For example, in kernel code, call `CUDAnative.sin`
instead of plain `sin`.

For a list of available functions, look at `src/device/cuda/libdevice.jl`.
