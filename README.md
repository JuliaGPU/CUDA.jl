CUDAdrv.jl
==========

*A Julia wrapper for the CUDA driver API.*

**Build status**: [![][buildbot-0.5-img]][buildbot-0.5-url] [![][buildbot-0.6-img]][buildbot-0.6-url] [![][buildbot-master-img]][buildbot-master-url]

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

**Code coverage**: [![][coverage-img]][coverage-url]

[buildbot-0.5-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAdrv.jl:%20Julia%200.5%20(x86-64)&badge=Julia%20v0.5
[buildbot-0.5-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAdrv.jl%3A%20Julia%200.5%20%28x86-64%29
[buildbot-0.6-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAdrv.jl:%20Julia%200.6%20(x86-64)&badge=Julia%20v0.6
[buildbot-0.6-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAdrv.jl%3A%20Julia%200.6%20%28x86-64%29
[buildbot-master-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAdrv.jl:%20Julia%20master%20(x86-64)&badge=Julia%20master
[buildbot-master-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAdrv.jl%3A%20Julia%20master%20%28x86-64%29

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://juliagpu.github.io/CUDAdrv.jl/stable
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://juliagpu.github.io/CUDAdrv.jl/latest

[coverage-img]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/coverage.svg
[coverage-url]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl

This package aims to provide high-level wrappers for the functionality exposed by the CUDA
driver API, and is meant for users who need high- or low-level access to the CUDA toolkit or
the underlying hardware.



Installation
------------

CUDAdrv is a registered package, and can be installed using the Julia package manager:

```julia
Pkg.add("CUDAdrv")
```

Julia versions 0.5 and 0.6 are supported, with limited effort to keep the package working on
current master as well. Refer to [the documentation][docs-stable-url] for more information
on how to install or use this package.
