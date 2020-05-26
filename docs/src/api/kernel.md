# Kernel programming

This section lists the package's public functionality that corresponds to special CUDA
functions for use in device code. It is loosely organized according to the [C language
extensions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions)
appendix from the CUDA C programming guide. For more information about certain intrinsics,
refer to the aforementioned NVIDIA documentation.


## Indexing and Dimensions

```@docs
gridDim
blockIdx
blockDim
threadIdx
warpsize
```


## Memory Types

### Shared Memory

```@docs
@cuStaticSharedMem
@cuDynamicSharedMem
```


## Synchronization

```@docs
sync_threads
sync_warp
threadfence_block
threadfence
threadfence_system
```

## Clock & Sleep

```@docs
clock
nanosleep
```

## Warp Vote

The warp vote functions allow the threads of a given warp to perform a
reduction-and-broadcast operation. These functions take as input a boolean predicate from
each thread in the warp and evaluate it. The results of that evaluation are combined
(reduced) across the active threads of the warp in one different ways, broadcasting a single
return value to each participating thread.

```@docs
vote_all
vote_any
vote_ballot
```


## Warp Shuffle

```@docs
shfl_sync
shfl_up_sync
shfl_down_sync
shfl_xor_sync
```


## Formatted Output

```@docs
@cuprint
@cuprintf
@cushow
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
CUDA.atomic_mul!
CUDA.atomic_div!
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


## CUDA runtime

Certain parts of the CUDA API are available for use on the GPU, for example to launch
dynamic kernels or set-up cooperative groups. Coverage of this part of the API, provided by
the `libcudadevrt` library, is under development and contributions are welcome.

Calls to these functions are often ambiguous with their host-side equivalents. To avoid
confusion, you need to prefix device-side API interactions with the CUDA module, e.g.,
`CUDA.synchronize`.

```@docs
CUDA.synchronize
```


## Math

Many mathematical functions are provided by the `libdevice` library, and are wrapped by
jl. These functions implement interfaces that are similar to existing functions
in `Base`, albeit often with support for fewer types.

To avoid confusion with existing implementations in `Base`, you need to prefix calls to this
library with the CUDA module. For example, in kernel code, call `CUDA.sin` instead of plain
`sin`.

For a list of available functions, look at `src/device/cuda/libdevice.jl`.


## Device array

CUDA.jl provides a primitive, lightweight array type to manage GPU data organized in an
plain, dense fashion. This is the device-counterpart to the `CuArray`, and implements (part
of) the array interface as well as other functionality for use _on_ the GPU:

```@docs
CuDeviceArray
CUDA.Const
```


## WMMA

Warp matrix multiply-accumulate (WMMA) is a CUDA API to access Tensor Cores, a new hardware
feature in Volta GPUs to perform mixed precision matrix multiply-accumulate operations. The
interface is split in two levels, both available in the WMMA submodule: low level wrappers
around the LLVM intrinsics, and a higher-level API similar to that of CUDA C.

!!! note
    Requires Julia v"1.4.0-DEV.666" or later, or you run into LLVM errors.

!!! note
    For optimal performance, you should use Julia `v1.5.0-DEV.324` or later.

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

The LLVM intrinsics are subdivided in three categories: load, store and multiply-accumulate.
In what follows, each of these will be discussed.

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

#### Example

````@eval
lines = readlines("../../../examples/wmma/low-level.jl")
start = findfirst(x -> x == "### START", lines) + 1
stop = findfirst(x -> x == "### END", lines) - 1
example = join(lines[start:stop], '\n')

using Markdown
Markdown.parse("""
```julia
$(example)
```
""")
````

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

#### Example

````@eval
lines = readlines("../../../examples/wmma/high-level.jl")
start = findfirst(x -> x == "### START", lines) + 1
stop = findfirst(x -> x == "### END", lines) - 1
example = join(lines[start:stop], '\n')

using Markdown
Markdown.parse("""
```julia
$(example)
```
""")
````
