# CUDA programming in Julia

The CUDA.jl package is the main entrypoint for for programming NVIDIA GPUs using CUDA. The
package makes it possible to do so at various abstraction levels, from easy-to-use arrays
down to hand-written kernels using low-level CUDA APIs.

If you have any questions, please feel free to use the `#gpu` channel on the [Julia
slack](https://julialang.slack.com/), or the [GPU domain of the Julia
Discourse](https://discourse.julialang.org/c/domain/gpu).


## Quick Start

The Julia CUDA stack requires a functional CUDA-setup, which includes both a driver and
matching toolkit. Once you've set that up, continue by installing the CUDA.jl package:

```julia
using Pkg
Pkg.add("CUDA")
```

To make sure everything works as expected, try to load the package and if you have the time
execute its test suite:

```julia
using CUDA

using Pkg
Pkg.test("CUDA")
```

For more details on the installation process, consult the [Installation](@ref
InstallationOverview) section. To understand the toolchain in more detail, have a look at
the tutorials in this manual. **It is highly recommended that new users start with the
[Introduction](@ref) tutorial**. For an overview of the available functionality, read the
[Usage](@ref UsageOverview) section. The following resources may also be of interest:

- Effectively using GPUs with Julia: [video](https://www.youtube.com/watch?v=7Yq1UyncDNc),
  [slides](https://docs.google.com/presentation/d/1l-BuAtyKgoVYakJSijaSqaTL3friESDyTOnU2OLqGoA/)
- How Julia is compiled to GPUs: [video](https://www.youtube.com/watch?v=Fz-ogmASMAE)


## Acknowledgements

The Julia CUDA stack has been a collaborative effort by many individuals. Significant
contributions have been made by the following individuals:

- Tim Besard (@maleadt) (lead developer)
- Valentin Churavy (@vchuravy)
- Mike Innes (@MikeInnes)
- Katharine Hyatt (@kshyatt)
- Simon Danisch (@SimonDanisch)


## Supporting and Citing

Much of the software in this ecosystem was developed as part of academic research. If you
would like to help support it, please star the relevant repository as such metrics may help
us secure funding in the future. If you use our software as part of your research, teaching,
or other activities, we would be grateful if you could cite our work. The
[CITATION.bib](https://github.com/JuliaGPU/CUDA.jl/blob/master/CITATION.bib) at the top of
this repository lists the relevant papers.
