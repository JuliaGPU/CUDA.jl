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
[codespeed-chart-url]: https://speed.juliagpu.org/timeline/#/?exe=9,11&env=1&base=none&ben=grid&revs=50

[codespeed-trend-img]: https://img.shields.io/badge/benchmarks-Trend-yellowgreen
[codespeed-trend-url]: https://speed.juliagpu.org/changes/?exe=9&env=1&tre=50

The CUDA.jl package is the main programming interface for working with NVIDIA CUDA GPUs
using Julia. It features a user-friendly array abstraction, a compiler for writing CUDA
kernels in Julia, and wrappers for various CUDA libraries.


## Quick start

Before all, make sure you have a recent NVIDIA driver. On Windows, also make sure you have
the [Visual C++ redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) installed.
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


## Requirements

The latest development version of CUDA.jl requires **Julia 1.8** or higher. If you are using
an older version of Julia, you need to use a previous version of CUDA.jl. This will happen
automatically when you install the package using Julia's package manager.

Note that CUDA.jl may not work with a custom build of Julia; it is recommended that you
install Julia using the [official binaries](https://julialang.org/downloads/) or
[juliaup](https://github.com/JuliaLang/juliaup).

The latest version of CUDA.jl also has certain requirements that cannot be enforced by the
package manager:

- Host platform: only 64-bit Linux and Windows are supported;
- Device hardware: only NVIDIA GPUs with **compute capability 3.5** (Kepler) or higher are
  supported;
- NVIDIA driver: a driver for **CUDA 11.0** or newer is required;
- CUDA toolkit (in case you need to use your own): only **CUDA toolkit 11.4** or newer are
  supported.

If you cannot meet these requirements, you may need to install an older version of CUDA.jl:

* CUDA.jl v5.3 is the last version with support for PowerPC (removed in v5.4)
* CUDA.jl v4.4 is the last version with support for CUDA 11.0-11.3 (deprecated in v5.0)
* CUDA.jl v4.0 is the last version to work with CUDA 10.2 (removed in v4.1)
* CUDA.jl v3.13 is the last version to work with CUDA 10.1 (removed in v4.0)
* CUDA.jl v1.3 is the last version to work with CUDA 9-10.0 (removed in v2.0)


## Supporting and Citing

Much of the software in this ecosystem was developed as part of academic research. If you
would like to help support it, please star the repository as such metrics may help us secure
funding in the future. If you use our software as part of your research, teaching, or other
activities, we would be grateful if you could cite our work. The
[CITATION.bib](https://github.com/JuliaGPU/CUDA.jl/blob/master/CITATION.bib) file in the
root of this repository lists the relevant papers.


## Project Status

The package is tested against, and being developed for, Julia 1.8 and above. Main
development and testing happens on x86 Linux, but the package is expected to work on
Windows and ARM and as well.


## Questions and Contributions

Usage questions can be posted on the [Julia Discourse
forum](https://discourse.julialang.org/c/domain/gpu) under the GPU domain and/or in the #gpu
channel of the [Julia Slack](https://julialang.org/community/).

Contributions are very welcome, as are feature requests and suggestions. Please open an
[issue](https://github.com/JuliaGPU/CUDA.jl/issues) if you encounter any problems.
