# CUDAdrv.jl

Code Coverage: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUDAdrv.jl)

This package wraps the [CUDA driver API](http://docs.nvidia.com/cuda/cuda-driver-api/). It
is meant for users who need low-level access to the CUDA toolkit or the underlying GPU. For
a wrapper of the higher-level [runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/),
see [CUDArt](https://github.com/JuliaGPU/CUDArt.jl).


Installation
------------

Requirements:

* Julia 0.5 or higher (use
  [v0.1.0](https://github.com/JuliaGPU/CUDAdrv.jl/releases/tag/v0.1.0) of this package
  for compatibility with Julia 0.4)
* CUDA toolkit: tested up until v8.0, newer versions might work but can be incompatible
* CUDA hardware

```
Pkg.add("CUDAdrv")
Pkg.test("CUDAdrv")
```


Debugging
---------

For extra information about what's happening behind the scenes, you can enable extra output
by defining the `DEBUG` environment variable, or `TRACE` for even more information.

Note that these features are incompatible with precompilation as the debugging code is
enabled statically in order to avoid any run-time overhead, so you need to wipe the compile
cache or run Julia using `--compilecache=no`. Enabling colors with `--color=yes` is also
recommended as it color-codes the output.
