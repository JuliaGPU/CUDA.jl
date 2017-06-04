# CUDAnative.jl

*Support for compiling and executing native Julia kernels on CUDA hardware.*

This package provides support for compiling and executing native Julia kernels on CUDA
hardware. It is a work in progress, and only works on very recent versions of Julia .


## Installation

Requirements:

* Julia 0.6 with LLVM 3.9 **built from source**, executed **in tree** (for LLVM.jl)
* NVIDIA driver, providing `libcuda.so` (for CUDAdrv.jl)
* CUDA toolkit

Although that first requirement might sound complicated, it basically means you need to
fetch and compile a copy of Julia 0.6 (refer to [the main repository's
README](https://github.com/JuliaLang/julia/blob/master/README.md#source-download-and-compilation),
checking out the latest tag for 0.6), and execute the resulting `julia` binary in-place
without doing a `make install`. Afterwards, you can do:

```
Pkg.add("CUDAnative")
Pkg.test("CUDAnative")
```

For now, only Linux and macOS are supported.


## Manual Outline

```@contents
Pages = [
    "man/usage.md",
    "man/troubleshooting.md",
    "man/performance.md"
]
```


## Library Outline

```@contents
Pages = [
    "lib/intrinsics.md"
]
```
