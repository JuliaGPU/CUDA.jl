# CUDA.jl

*CUDA programming in Julia*

This repository hosts a Julia package that bundles functionality from several other packages
for CUDA programming, and provides high-level documentation and tutorials for effectively
using CUDA GPUs from Julia. The documentation is accessible at
[juliagpu.gitlab.io](https://juliagpu.gitlab.io/CUDA.jl/).


CUDA.jl includes functionality from the following packages:

- [CUDAdrv.jl](https://github.com/JuliaGPU/CUDAdrv.jl): interface to the CUDA driver
- [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl): kernel programming capabilities
- [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl): GPU array abstraction

For details on the APIs that these packages expose, refer to the associated documentation.


## API stability

Versioning of this package follows [SemVer](https://semver.org/) as used by the Julia
package manager: Depending on a specific major version of CUDA.jl should guarantee that your
application will not break, as long as it only uses functionality from the package's public
API. For CUDA.jl, this API includes certain non-exported functions and macros that would
otherwise clash with implementations in Julia. Refer to [src/CUDA.jl](src/CUDA.jl) for more
details.
