# CUDA.jl

*CUDA programming in Julia*

| **Documentation**                                                         | **Build Status**                                                    | **Performance**                     |
|:-------------------------------------------------------------------------:|:-------------------------------------------------------------------:|:-----------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][buildkite-img]][buildkite-url] [![][codecov-img]][codecov-url] | [![][codespeed-img]][codespeed-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliagpu.github.io/CUDA.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliagpu.github.io/CUDA.jl/dev/

[buildkite-img]: https://badge.buildkite.com/27aaeb352a9420297ed2d30cb055ac383a399ea8f121599912.svg?branch=master
[buildkite-url]: https://buildkite.com/julialang/cuda-dot-jl

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDA.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDA.jl

[codespeed-img]: https://img.shields.io/badge/codespeed-Benchmarks-informational
[codespeed-url]: https://speed.juliagpu.org/timeline/#/?exe=6&ben=grid&env=1&revs=50

The CUDA.jl package is the main programming interface for working with NVIDIA CUDA GPUs
using Julia. It features a user-friendly array abstraction, a compiler for writing CUDA
kernels in Julia, and wrappers for various CUDA libraries.


## Requirements

The latest development version of CUDA.jl requires **Julia 1.6** or higher. If you are using
an older version of Julia, you need to use a released version of CUDA.jl. This will happen
automatically when you install the package using Julia's package manager.

CUDA.jl currently also requires a CUDA-capable GPU with **compute capability 5.0** (Maxwell)
or higher, and an accompanying NVIDIA driver with support for **CUDA 10.1** or newer. These
requirements are not enforced by the Julia package manager when installing CUDA.jl.
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

For an overview of the CUDA toolchain in use, you can run the following command after
importing the package:

```julia
julia> using CUDA

julia> CUDA.versioninfo()
```

This may take a while, as it will precompile the package and download a suitable version of
the CUDA toolkit. If you prefer to use your own (not recommended), set the
`JULIA_CUDA_USE_BINARYBUILDER` environment variable to `false` before importing the package.

If your GPU is not fully supported, the above command (or any other command that initializes
the toolkit) will issue a warning. Your devices' compute capability will be listed as part of
the `versioninfo()` output, but you can always query it explicitly:

```julia
julia> [CUDA.capability(dev) for dev in CUDA.devices()]
1-element Vector{VersionNumber}:
 v"5.0.0"
```

For more usage instructions and other information, please refer to [the
documentation](https://juliagpu.github.io/CUDA.jl/stable/).


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

