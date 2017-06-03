CUDAdrv.jl
==========

**Build status**: [![][buildbot-0.5-img]][buildbot-0.5-url] [![][buildbot-0.6-img]][buildbot-0.6-url] [![][buildbot-master-img]][buildbot-master-url]

**Documentation**: [![][docs-latest-img]][docs-latest-url]

**Code coverage**: [![][coverage-img]][coverage-url]

[buildbot-0.5-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAdrv.jl:%20Julia%200.5%20(x86-64)&badge=Julia%20v0.5
[buildbot-0.5-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAdrv.jl%3A%20Julia%200.5%20%28x86-64%29
[buildbot-0.6-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAdrv.jl:%20Julia%200.6%20(x86-64)&badge=Julia%200.6
[buildbot-0.6-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAdrv.jl%3A%20Julia%200.6%20%28x86-64%29
[buildbot-master-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAdrv.jl:%20Julia%20master%20(x86-64)&badge=Julia%20master
[buildbot-master-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAdrv.jl%3A%20Julia%20master%20%28x86-64%29

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://juliagpu.github.io/CUDAdrv.jl/latest

[coverage-img]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/coverage.svg
[coverage-url]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl

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

If you get an error `ERROR_NO_DEVICE` (`No CUDA-capable device`) upon loading CUDAdrv.jl,
CUDA could not detect any capable GPU. It probably means that your GPU isn't supported by
the CUDA/NVIDIA driver loaded by CUDAdrv.jl, or that your set-up is damaged in some way.
Please make sure that (1) your GPU is supported by the current driver (you might need the
so-called legacy driver, refer to the CUDA installation instructions for your platform), and
(2) CUDAdrv.jl targets the correct driver library (check the `libcuda_path` variable in
`CUDAdrv/deps/ext.jl`, or run `Pkg.build` with the `DEBUG` environment variable set to 1).



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
