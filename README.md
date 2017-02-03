CUDAdrv.jl
==========

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
* NVIDIA driver, providing `libcuda.so` (the full CUDA toolkit is not necessary)
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
with a `INVALID_CONTEXT` or `CONTEXT_IS_DESTROYED` error message.

If this seems to be the case, re-run with `TRACE=1` and file a bug report.



Troubleshooting
---------------

You can enable verbose logging using two environment variables:

* `DEBUG`: if set, enable additional (possibly costly) run-time checks, and some more
  verbose output
* `TRACE`: if set, the `DEBUG` level will be activated, in addition with a trace of every
  call to the underlying library

In order to avoid run-time cost for checking the log level, these flags are implemented by
means of global constants. As a result, you **need to run Julia with precompilation
disabled** if you want to modify these flags:

```
$ TRACE=1 julia --compilecache=no examples/vadd.jl
TRACE: CUDAdrv.jl is running in trace mode, this will generate a lot of additional output
...
```

Enabling colors with `--color=yes` is also recommended as it color-codes the output.
