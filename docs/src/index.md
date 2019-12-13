# CUDA programming in Julia

Julia has several packages for programming NVIDIA GPUs using CUDA. Some of these packages
focus on performance and flexibility, while others aim to raise the abstraction level and
improve performance. This website will introduce the different options, how to use them, and
what best to choose for your application. For more specific details, such as API references
or development practices, refer to each package's own documentation.

If you have any questions, please feel free to use the `#gpu` channel on the [Julia
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

Before installing any package, you need to make sure you have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx) and matching [CUDA
toolkit](https://developer.nvidia.com/cuda-downloads). On Linux, you can verify driver
availability by executing `nvidia-smi`, and you have installed CUDA successfully if you can
execute `ptxas --version`. If possible, you should install these dependencies using a
package manager instead of downloading them from the NVIDIA homepage; refer to your
distribution's documentation for more details.

Once CUDA has been set up, continue by installing the three core packages that make up the
Julia CUDA stack:

```julia
using Pkg
Pkg.add(["CUDAdrv", "CUDAnative", "CuArrays"])
```

To make sure everything works as expected, try to load the packages and if you have the time
execute their test suites:

```julia
using CUDAdrv, CUDAnative, CuArrays

using Pkg
Pkg.test(["CUDAdrv", "CUDAnative", "CuArrays"])
```

If you want to use these packages only when a GPU is available, consult the [Conditional
Usage](@ref) section.

!!! note

    In the future, the public functionality from these packages will be bundled in a single
    package, CUDA.jl.

To understand the toolchain in more detail, check out the tutorials in this manual. **It is
highly recommended that new users start with the [Introduction](@ref) tutorial**. For an
overview of the available functionality, check the [Overview](@ref) page.


## Acknowledgements

The Julia CUDA stack has been a collaborative effort by many individuals. Significant
contributions have been made by the following individuals:

- Tim Besard (@maleadt) (lead developer)
- Valentin Churavy (@vchuravy)
- Mike Innes (@MikeInnes)
- Katharine Hyatt (@kshyatt)
- Simon Danisch (@SimonDanisch)
