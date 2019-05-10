CUDAnative.jl
=============

*Support for compiling and executing native Julia kernels on CUDA hardware.*

[![][docs-latest-img]][docs-latest-url] [![][codecov-img]][codecov-url] [![][doi-img]][doi-url]

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDAnative.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDAnative.jl

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://juliagpu.gitlab.io/CUDAnative.jl/

[doi-img]: https://zenodo.org/badge/DOI/10.1109/TPDS.2018.2872064.svg
[doi-url]: https://doi.org/10.1109/TPDS.2018.2872064



Installation
------------

CUDAnative is a registered package, and can be installed using the Julia package manager:

```julia
Pkg.add("CUDAnative")
```

**NOTE**: the current version of this package requires Julia 1.0. Only older versions of this package, v0.6.x or older, work with Julia 0.6, and require a source-build of Julia.


License
-------

CUDAnative.jl is licensed under the [MIT license](LICENSE.md).

If you use this package in your research, please cite the paper [Besard, Foket,
De Sutter (2018)](https://doi.org/10.1109/TPDS.2018.2872064).  For your
convenience, a BibTeX entry is provided in the [`CITATION.bib`](CITATION.bib)
file.
