# CUDAdrv.jl

*A Julia wrapper for the CUDA driver API.*

This package aims to provide high-level wrappers for the functionality exposed by the CUDA
driver API, and is meant for users who need high- or low-level access to the CUDA toolkit or
the underlying hardware.

The package is built upon the [low-level CUDA driver
API](http://docs.nvidia.com/cuda/cuda-driver-api/), but that shouldn't make the Julia
wrapper any harder to use. That said, it is a work-in-progress and does not offer the same
functionality or convenience as the more popular
[CUDArt](https://github.com/JuliaGPU/CUDArt.jl) package, which is built upon the
[higher-level CUDA runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/).


## Installation

Requirements:

* Julia 0.5 or higher (use
  [v0.1.0](https://github.com/JuliaGPU/CUDAdrv.jl/releases/tag/v0.1.0) of this package
  for compatibility with Julia 0.4)
* NVIDIA driver, providing `libcuda.so` (the full CUDA toolkit is not required)
* CUDA hardware

At the Julia REPL:

```julia
Pkg.add("CUDAdrv")
using CUDAdrv

# optionally
Pkg.test("CUDAdrv")
```

Building CUDAdrv might display error messages, indicating issues with your set-up. These
messages can be cryptic as they happen too early for decent error handling to be loaded.
However, please pay close attention to them as they might prevent CUDAdrv.jl from working
properly! Some common issues:

* unknown error (code 999): this often indicates that your set-up is broken, eg. because you
  didn't load the correct, or any, kernel module. Please verify your set-up, on Linux by
  executing `nvidia-smi` or on other platforms by compiling and running CUDA C code using
  `nvcc`.
* no device (code 100): CUDA didn't detect your device, because it is not supported by CUDA
  or because you loaded the wrong kernel driver (eg. legacy when you need regular, or
  vice-versa). CUDAdrv.jl cannot work in this case, because CUDA does not allow us to query
  the driver version without a valid device, something we need in order to version the API
  calls.
* using library stubs (code -1): if any API call returns -1, you're probably using the CUDA
  driver library stubs which return this value for every function call. This is not
  supported by CUDAdrv.jl, and is only intended to be used when compiling C or C++ code to
  be linked with `libcuda.so` at a time when that library isn't available yet. Unless you
  purposefully added the stub libraries to the search path, please run the build script with
  `JULIA_DEBUG=CUDAdrv` and file a bug report.

Even if the build fails, CUDAdrv.jl should always be loadable. This simplifies use by
downstream packages, until there is proper language support for conditional modules. You can
check whether the package has been built properly by inspecting the `CUDAdrv.configured`
global variable.
