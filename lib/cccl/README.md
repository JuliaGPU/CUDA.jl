# CCCL.jl

Julia wrapper for the C parallel library of [NVIDIA CCCL](https://github.com/NVIDIA/cccl)
(`cccl.c.parallel.v2`, "HostJIT"), providing CUB's device-wide algorithms (reduce, scan,
segmented reduce, radix sort, merge sort) with operators written in Julia.

Every combination of algorithm, element types and operator is JIT-compiled on first use.
Operators are compiled to LLVM IR with GPUCompiler and linked into the CUB kernels, where
they inline. Builds are cached for the rest of the session; precompiled CUB headers are
cached on disk, so later sessions build faster.

**This package is a prototype.** It depends on `CCCL_C_jll` 3.6, the first version
providing the HostJIT library, which is only built for Linux x86_64. HostJIT compiles
against the CUDA runtime headers and links against the CUDA runtime library, which it takes
from the CUDA JLLs that CUDA.jl uses if those ship every header it needs. Current JLLs lack
the compiler-support headers (`include/crt`), so HostJIT needs a local CUDA toolkit for now
(see `CCCL.toolkit_path`). Either way, nvJitLink needs to be from CUDA 13.5 or a later 13.x.
Otherwise, `CCCL.functional()` returns false.

```julia
using CUDACore, CCCL

x = CuArray(rand(Float32, 1024))
CCCL.reduce(+, x; init=0f0)
CCCL.reduce((a, b) -> max(abs(a), abs(b)), x; init=0f0)

y = similar(x)
CCCL.inclusive_scan!(+, y, x)

CCCL.radix_sort!(x)
CCCL.merge_sort!(x; lt=(a, b) -> abs(a - 0.5f0) < abs(b - 0.5f0))
```

This package is part of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) ecosystem.
