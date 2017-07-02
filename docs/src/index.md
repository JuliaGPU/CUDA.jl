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
using CUDAnative

# optionally
Pkg.test("CUDAnative")
```

For now, only Linux and macOS are supported. The build step will discover the available CUDA
and LLVM installations, and figure out which devices can be programmed using that set-up. It
depends on CUDAdrv and LLVM being properly configured.

Even if the build fails, CUDAnative.jl should always be loadable. This simplifies use by
downstream packages, until there is proper language support for conditional modules. You can
check whether the package has been built properly by inspecting the `CUDAnative.configured`
global variable.
