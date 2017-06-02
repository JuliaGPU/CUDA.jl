# CUDAdrv.jl

A Julia wrapper for the CUDA toolkit.

This package aims to provide high-level wrappers for the functionality exposed by the CUDA
driver API, and is meant for users who need high- or low-level access to the CUDA toolkit or
the underlying GPU.

The package is built upon the [low-level CUDA driver
API](http://docs.nvidia.com/cuda/cuda-driver-api/), but that shouldn't make the Julia
wrapper any harder to use. That said, it is a work-in-progress and does not offer the same
functionality or convenience as the more popular
[CUDArt](https://github.com/JuliaGPU/CUDArt.jl) package, which is built upon the
[higher-level CUDA runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/).


## Installation

At the Julia REPL:

```julia
Pkg.add("CUDAdrv")
```
