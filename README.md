# CUDA.jl

*CUDA programming in Julia*

| **Documentation**                     | **Build Status**                                              |
|:-------------------------------------:|:-------------------------------------------------------------:|
| [![][docs-latest-img]][docs-latest-url] | [![][gitlab-img]][gitlab-url] [![][codecov-img]][codecov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://juliagpu.gitlab.io/CUDA.jl/

[gitlab-img]: https://gitlab.com/JuliaGPU/CUDA.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CUDA.jl/commits/master

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDA.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDA.jl

The CUDA.jl package is the main programming interface for working with NVIDIA CUDA GPUs
using Julia. It features a user-friendly array abstraction, a compiler for writing CUDA
kernels in Julia, and wrappers for various CUDA libraries.


## Requirements

The current version of CUDA.jl requires **Julia 1.5** or higher, a CUDA-capable GPU with
**compute capability 5.0** (Maxwell) or higher, and an accompanying NVIDIA driver with
support for **CUDA 10.1** or newer.

These requirements are not enforced by the Julia package manager when installing CUDA.jl.
Depending on your system and GPU, you may need to install an older version of the package.


## Quick start

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add CUDA
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("CUDA")
```

For usage instructions and other information, please refer to the documentation at
[juliagpu.gitlab.io](https://juliagpu.gitlab.io/CUDA.jl/).


## Supporting and Citing

Much of the software in this ecosystem was developed as part of academic research. If you
would like to help support it, please star the repository as such metrics may help us secure
funding in the future. If you use our software as part of your research, teaching, or other
activities, we would be grateful if you could cite our work. The
[CITATION.bib](https://github.com/JuliaGPU/CUDA.jl/blob/master/CITATION.bib) file in the
root of this repository lists the relevant papers.


## Project Status

The package is tested against, and being developed for, Julia 1.3 and above. Main
development and testing happens on Linux, but the package is expected to work on macOS and
Windows as well.


## Questions and Contributions

Usage questions can be posted on the [Julia Discourse
forum](https://discourse.julialang.org/c/domain/gpu) under the GPU domain and/or in the #gpu
channel of the [Julia Slack](https://julialang.org/community/).

Contributions are very welcome, as are feature requests and suggestions. Please open an
[issue](https://github.com/JuliaGPU/CUDA.jl/issues) if you encounter any problems.

