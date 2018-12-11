CUDAdrv.jl
==========

*A Julia wrapper for the CUDA driver API.*

[![][docs-latest-img]][docs-latest-url] [![][codecov-img]][codecov-url] [![][doi-img]][doi-url]

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://juliagpu.gitlab.io/CUDAdrv.jl/

[doi-img]: https://zenodo.org/badge/DOI/10.1109/TPDS.2018.2872064.svg
[doi-url]: https://doi.org/10.1109/TPDS.2018.2872064

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
paper](https://ieeexplore.ieee.org/document/8471188):

```
@article{besard:2017,
  author    = {Besard, Tim and Foket, Christophe and De Sutter, Bjorn},
  title     = {Effective Extensible Programming: Unleashing {Julia} on {GPUs}},
  journal   = {IEEE Transactions on Parallel and Distributed Systems},
  year      = {2018},
  doi       = {10.1109/TPDS.2018.2872064},
  ISSN      = {1045-9219},
}
```
