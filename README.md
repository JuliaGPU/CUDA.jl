# CUDA.jl

*CUDA programming in Julia*

[![][doi-img]][doi-url] [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] [![][buildkite-img]][buildkite-url] [![][codecov-img]][codecov-url] [![][codespeed-trend-img]][codespeed-trend-url] [![][codespeed-chart-img]][codespeed-chart-url]

[doi-img]: https://zenodo.org/badge/doi/10.1109/TPDS.2018.2872064.svg
[doi-url]: https://ieeexplore.ieee.org/abstract/document/8471188

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://cuda.juliagpu.org/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://cuda.juliagpu.org/dev/

[buildkite-img]: https://badge.buildkite.com/27aaeb352a9420297ed2d30cb055ac383a399ea8f121599912.svg?branch=master
[buildkite-url]: https://buildkite.com/julialang/cuda-dot-jl

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDA.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDA.jl

[codespeed-chart-img]: https://img.shields.io/badge/benchmarks-Chart-yellowgreen
[codespeed-chart-url]: https://speed.juliagpu.org/timeline/#/?exe=6&env=1&base=none&ben=grid&revs=50

[codespeed-trend-img]: https://img.shields.io/badge/benchmarks-Trend-yellowgreen
[codespeed-trend-url]: https://speed.juliagpu.org/changes/?exe=6&env=1&tre=50

The CUDA.jl package is the main programming interface for working with NVIDIA CUDA GPUs
using Julia. It features a user-friendly array abstraction, a compiler for writing CUDA
kernels in Julia, and wrappers for various CUDA libraries.


## Requirements

The latest development version of CUDA.jl requires **Julia 1.6** or higher. If you are using
an older version of Julia, you need to use a previous version of CUDA.jl. This will happen
automatically when you install the package using Julia's package manager.

CUDA.jl currently also requires a CUDA-capable GPU with **compute capability 3.5** (Kepler)
or higher, and an accompanying NVIDIA driver with support for **CUDA 11.0** or newer. These
requirements are not enforced by the Julia package manager when installing CUDA.jl.
Depending on your system and GPU, you may need to install an older version of the package.


## Quick start

Before all, make sure you have a recent NVIDIA driver. On Windows, make sure you have the
[Visual C++ redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) installed.
You do not need to install the CUDA Toolkit.

CUDA.jl can be installed with the Julia package manager. From the Julia REPL, type `]` to
enter the Pkg REPL mode and run:

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
the CUDA toolkit. If your GPU is not fully supported, the above command (or any other
command that initializes the toolkit) will issue a warning.

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

The package is tested against, and being developed for, Julia 1.6 and above. Main
development and testing happens on x86 Linux, but the package is expected to work on
Windows, and on ARM and PowerPC as well.


## Questions and Contributions

Usage questions can be posted on the [Julia Discourse
forum](https://discourse.julialang.org/c/domain/gpu) under the GPU domain and/or in the #gpu
channel of the [Julia Slack](https://julialang.org/community/).

Contributions are very welcome, as are feature requests and suggestions. Please open an
[issue](https://github.com/JuliaGPU/CUDA.jl/issues) if you encounter any problems.
