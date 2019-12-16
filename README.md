# CUDAdrv.jl

*A Julia wrapper for the CUDA driver API.*

| **Documentation**                                                       | **Build Status**                                              |
|:-----------------------------------------------------------------------:|:-------------------------------------------------------------:|
| [![][docs-usage-img]][docs-usage-url] [![][docs-api-img]][docs-api-url] | [![][gitlab-img]][gitlab-url] [![][codecov-img]][codecov-url] |

[gitlab-img]: https://gitlab.com/JuliaGPU/CUDAdrv.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CUDAdrv.jl/commits/master

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl

[docs-usage-img]: https://img.shields.io/badge/docs-usage-blue.svg
[docs-usage-url]: https://juliagpu.gitlab.io/CUDA.jl/

[docs-api-img]: https://img.shields.io/badge/docs-api-blue.svg
[docs-api-url]: https://juliagpu.gitlab.io/CUDAdrv.jl/

CUDAdrv.jl proviudes high-level wrappers for the functionality exposed by the CUDA driver
API, and is meant for users who need high- or low-level access to the CUDA toolkit or the
underlying hardware.


## Quick start

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add CUDAdrv
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("CUDAdrv")
```

For usage instructions and other information, check-out the [CUDA.jl
documentation](https://juliagpu.gitlab.io/CUDA.jl/).


## Project Status

The package is tested against, and being developed for, Julia `1.0` and above. Main
development and testing happens on Linux, but the package is expected to work on macOS and
Windows as well.


## Questions and Contributions

Usage questions can be posted on the [Julia Discourse
forum](https://discourse.julialang.org/c/domain/gpu) under the GPU domain and/or in the #gpu
channel of the [Julia Slack](https://julialang.org/community/).

Contributions are very welcome, as are feature requests and suggestions. Please open an
[issue](https://github.com/JuliaGPU/CUDAdrv.jl/issues) if you encounter any problems.
