CUDAnative.jl
=============

*Support for compiling and executing native Julia kernels on CUDA hardware.*

**Build status**: [![][buildbot-julia06-img]][buildbot-julia06-url] [![][buildbot-juliadev-img]][buildbot-juliadev-url]

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

**Code coverage** (of host code): [![][coverage-img]][coverage-url]

[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAnative-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAnative-julia06-x86-64bit
[buildbot-juliadev-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAnative-juliadev-x86-64bit&name=julia%20dev
[buildbot-juliadev-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAnative-juliadev-x86-64bit

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
Pkg.add("CUDAnative")
```

The package only works with Julia 0.6, which **you need to have built from source**. Refer
to [the documentation][docs-stable-url] for more information on how to install or use this
package.


License
-------

CUDAnative.jl is licensed under the [MIT License](LICENSE.md). The original author is Tim Besard.

If you use this package in your research, please cite the [2017 arXiv paper](https://arxiv.org/abs/1712.03112).
