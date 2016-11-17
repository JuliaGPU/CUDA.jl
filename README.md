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


Features
--------

In general, this wrapper tries to stay close to the abstraction level of the CUDA driver
API. However, there are some additional features:

### Automatic memory management

Except for the encapsulating context, `destroy` or `unload` calls are never needed. Objects
are registered with the Julia garbage collector, and are automatically finalized when they
go out of scope.

However, many CUDA API functions implicitly depend on global state, such as the current
active context. The wrapper needs to model those dependencies in order for objects not to
get destroyed before any dependent object is. If we fail to model these dependency
relations, API calls might randomly fail, eg. in the case of a missing context dependency
with a `INVALID_CONTEXT` error message.

If this seems to be the case, re-run with `TRACE=1` and file a bug report.


Debugging
---------

For extra information about what's happening behind the scenes, you can enable extra output
by defining the `DEBUG` environment variable, or `TRACE` for even more information.

Note that these features are incompatible with precompilation as the debugging code is
enabled statically in order to avoid any run-time overhead, so you need to wipe the compile
cache or run Julia using `--compilecache=no`. Enabling colors with `--color=yes` is also
recommended as it color-codes the output.
