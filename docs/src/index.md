# CUDA programming in Julia with CUDA.jl

Julia has several packages for high-performance, high-level GPU programming with NVIDIA CUDA
hardware. Some of these packages give uncompromising access to the low-level CUDA
programming interfaces and development tools, while others aim to raise the abstraction
level and even make it possible to use GPUs without specific programming experience.

The CUDA.jl package bundles all this functionality, and this site provides documentation on
how to use each part. It is is a work in progress, so any feedback or contributions are
appreciated and can be made on the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) repository
page. For specific API documentation, refer to the documentation of each package. If you
have any other questions, please feel free to use the `#gpu` channel on the [Julia
slack](https://julialang.slack.com/), or the [GPU domain of the Julia
Discourse](https://discourse.julialang.org/c/domain/gpu).


## Supporting and Citing

Much of the software in this ecosystem was developed as part of academic research. If you
would like to help support it, please star the relevant repository as such metrics may help
us secure funding in the future. If you use our software as part of your research, teaching,
or other activities, we would be grateful if you could cite our work. The
[CITATION.bib](https://github.com/JuliaGPU/CUDA.jl/blob/master/CITATION.bib) at the top of
this repository lists the relevant papers.


## Getting Started

### Installation

Most users will want to install and use the CUDA.jl package, which reexports
functionality from several other Julia packages:

```julia
using Pkg
Pkg.add("CUDA")
```

To load the package, use the command:

```julia
using CUDA
```

To understand the toolchain in more detail, check out the following tutorials in this
manual. **It is highly recommended that new users start with the [Introduction](@ref)
tutorial**. For an overview of the available functionality, check the [Overview](@ref) page.


### Tutorials

The following tutorials will introduce you to CUDA GPU programming in Julia.

```@contents
Pages = [
    "tutorials/introduction.md",
    ]
Depth = 1
```

### Usage

These pages introduce you to the essentials of CUDA programming in Julia, and the different
packages that make up the toolchain. It explains the general workflow, options which are
generally available, and the development tools you can use.

```@contents
Pages = [
    "usage/overview.md",
    ]
Depth = 1
```


## Acknowledgements

The Julia CUDA stack has been a collaborative effort by many individuals. Significant
contributions have been made by the following individuals:

- Tim Besard (@maleadt) (lead developer)
- Valentin Churavy (@vchuravy)
- Mike Innes (@MikeInnes)
- Katharine Hyatt (@kshyatt)
- Simon Danisch (@SimonDanisch)
