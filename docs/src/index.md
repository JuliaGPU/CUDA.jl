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

If you get an error `ERROR_NO_DEVICE` (`No CUDA-capable device`) upon loading CUDAdrv.jl,
CUDA could not detect any capable GPU. It probably means that your GPU isn't supported by
the CUDA/NVIDIA driver loaded by CUDAdrv.jl, or that your set-up is damaged in some way.
Please make sure that (1) your GPU is supported by the current driver (you might need the
so-called legacy driver, refer to the CUDA installation instructions for your platform), and
(2) CUDAdrv.jl targets the correct driver library (check the `libcuda_path` variable in
`CUDAdrv/deps/ext.jl`, or run `Pkg.build` with the `DEBUG` environment variable set to 1).


## Manual Outline

```@contents
Pages = [
    "man/usage.md"
]
```


## Library Outline

```@contents
Pages = [
    "lib/api.md",
    "lib/array.md"
]
```
