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
Pkg.test("CUDAdrv")
```

CUDAdrv can throw errors during build. These errors tend to be cryptic, as the library isn't
loaded yet and we can't look-up the string for a given error code. Common errors include:

* Code 999 (`UNKNOWN_ERROR`): this often indicates that your set-up is broken, eg. because
  you didn't load the correct, or any, kernel module. Please verify your set-up, on Linux by
  executing `nvidia-smi` or on other platforms by compiling and running CUDA C code using
  `nvcc`.
* Code 100 (`ERROR_NO_DEVICE`): CUDA didn't detect your device, because it is not supported
  by CUDA or because you loaded the wrong kernel driver (eg. legacy when you need regular,
  or vice-versa). CUDAdrv cannot work in this case, because CUDA does not allow us to query
  the driver version without a valid device, something we need in order to version the API
  calls.
* Code -1 (not a valid CUDA error): this error shouldn't ever be thrown by any CUDA API
  function. Possibly, CUDAdrv might have selected the library stubs instead, which always
  return -1. Please run the build script with `DEBUG=1` and file a bug report.
