# CUDAnative.jl

*Support for compiling and executing native Julia kernels on CUDA hardware.*

This package provides support for compiling and executing native Julia kernels on CUDA
hardware. It is a work in progress, and only works on very recent versions of Julia .


## Installation

Requirements:

* Julia 1.0
* CUDA toolkit
* NVIDIA driver

```
Pkg.add("CUDAnative")
using CUDAnative

# optionally
Pkg.test("CUDAnative")
```


## License

CUDAnative.jl is licensed under the [MIT
license](https://github.com/JuliaGPU/CUDAnative.jl/blob/master/LICENSE.md).

If you use this package in your research, please cite the paper [Besard, Foket,
De Sutter (2018)](https://doi.org/10.1109/TPDS.2018.2872064).  For your
convenience, a BibTeX entry is provided in the
[`CITATION.bib`](https://github.com/JuliaGPU/CUDAnative.jl/blob/master/CITATION.bib)
file.
