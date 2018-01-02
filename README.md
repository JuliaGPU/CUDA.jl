CUDAdrv.jl
##########

*A Julia wrapper for the CUDA driver API.*

**Build status**: [![][buildbot-julia06-img]][buildbot-julia06-url] [![][buildbot-juliadev-img]][buildbot-juliadev-url]

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

**Code coverage**: [![][coverage-img]][coverage-url]

[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAdrv-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAdrv-julia06-x86-64bit
[buildbot-juliadev-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAdrv-juliadev-x86-64bit&name=julia%20dev
[buildbot-juliadev-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAdrv-juliadev-x86-64bit

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


License
-------

CUDAdrv.jl is licensed under the [MIT license](LICENSE.md).

If you use this package in your research, please cite the [following
paper](https://arxiv.org/abs/1712.03112):

```
@article{besard:2017,
  author    = {Tim Besard and Christophe Foket and De Sutter, Bjorn},
  title     = {Effective Extensible Programming: Unleashing {Julia} on {GPUs}},
  journal   = {arXiv},
  volume    = {abs/11712.03112},
  year      = {2017},
  url       = {http://arxiv.org/abs/1712.03112},
}
```
