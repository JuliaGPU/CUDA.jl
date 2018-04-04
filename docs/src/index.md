# CUDAnative.jl

*Support for compiling and executing native Julia kernels on CUDA hardware.*

This package provides support for compiling and executing native Julia kernels on CUDA
hardware. It is a work in progress, and only works on very recent versions of Julia .


## Installation

Requirements:

* Julia 0.7
* CUDA toolkit
* NVIDIA driver

```
Pkg.add("CUDAnative")
using CUDAnative

# optionally
Pkg.test("CUDAnative")
```

The build step will discover the available CUDA and LLVM installations, and
figure out which devices can be programmed using that set-up. It depends on
CUDAdrv and LLVM being properly configured.

Even if the build fails, CUDAnative.jl should always be loadable. This simplifies use by
downstream packages, until there is proper language support for conditional modules. You can
check whether the package has been built properly by inspecting the `CUDAnative.configured`
global variable.
