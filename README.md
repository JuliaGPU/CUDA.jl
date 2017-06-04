CUDAnative.jl
=============

*Support for compiling and executing native Julia kernels on CUDA hardware.*

**Build status**: [![][buildbot-0.6-img]][buildbot-0.6-url] [![][buildbot-master-img]][buildbot-master-url]

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

**Code coverage** (of host code): [![][coverage-img]][coverage-url]

[buildbot-0.6-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAnative.jl:%20Julia%200.6%20(x86-64)&badge=Julia%20v0.6
[buildbot-0.6-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAnative.jl%3A%20Julia%200.6%20%28x86-64%29
[buildbot-master-img]: https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUDAnative.jl:%20Julia%20master%20(x86-64)&badge=Julia%20master
[buildbot-master-url]: https://ci.maleadt.net/buildbot/julia/builders/CUDAnative.jl%3A%20Julia%20master%20%28x86-64%29

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://juliagpu.github.io/CUDAnative.jl/stable
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://juliagpu.github.io/CUDAnative.jl/latest

[coverage-img]: https://codecov.io/gh/JuliaGPU/CUDAnative.jl/coverage.svg
[coverage-url]: https://codecov.io/gh/JuliaGPU/CUDAnative.jl


Installation
------------

CUDAnative is a registered package, and can be installed using the Julia package manager:

```julia
Pkg.add("CUDAdrv")
```

The package only works with Julia 0.6, which **you need to have built from source**. Refer
to [the documentation][docs-stable-url] for more information on how to install or use this
package.
