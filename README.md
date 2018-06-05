CUDAdrv.jl
==========

*A Julia wrapper for the CUDA driver API.*

**Build status**: [![pipeline status](https://gitlab.com/JuliaGPU/CUDAdrv.jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/CUDAdrv.jl/commits/master)

**Code coverage**: [![coverage report](https://gitlab.com/JuliaGPU/CUDAdrv.jl/badges/master/coverage.svg)](https://gitlab.com/JuliaGPU/CUDAdrv.jl/commits/master)

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://juliagpu.github.io/CUDAdrv.jl/stable
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://juliagpu.github.io/CUDAdrv.jl/latest

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
