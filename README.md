CUDAapi.jl
==========

*Reusable components for CUDA development.*

| **Build Status**                                                                                                                     | **Coverage**                    |
|:------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|
| [![][gitlab-img]][gitlab-url] [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![][codecov-img]][codecov-url] |

[gitlab-img]: https://gitlab.com/JuliaGPU/CUDAapi.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CUDAapi.jl/commits/master

[travis-img]: https://api.travis-ci.org/JuliaGPU/CUDAapi.jl.svg?branch=master
[travis-url]: https://travis-ci.org/JuliaGPU/CUDAapi.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/e41yic5p5ru018mf/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/maleadt/cudaapi-jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/CUDAapi.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/CUDAapi.html

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDAapi.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDAapi.jl

This package provides some reusable functionality for working with CUDA or
NVIDIA APIs. It is intended for package developers, and does not provide
concrete application functionality.


Usage
-----

### Availability

To check if a CUDA GPU is available, CUDAapi provides and exports the `has_cuda`
and `has_cuda_gpu` functions. These functions are useful to query whether you
will be able to import a package that requires CUDA to be available, such as
CuArrays.jl (CUDAapi.jl itself will always import without an error):

```julia
using CUDAapi # this will NEVER fail
if has_cuda()
    try
        using CuArrays # we have CUDA, so this should not fail
    catch ex
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    end
end
```


### Discovery

The file `src/discovery.jl` defines helper methods for discovering the NVIDIA
driver and CUDA toolkit, as well as some more generic methods to find libraries
and binaries relative to, e.g., the location of the driver or toolkit.


### Compatibility

The file `src/compatibility.jl` contains hard-coded databases with software and hardware
compatibility information that cannot be queried from APIs.



Maintenance
-----------

### CUDA version update

When a new version of CUDA is released, CUDAapi.jl needs to be updated
accordingly:

- `discovery.jl`: update the `cuda_versions` dictionary
- `compatibility.jl`: update each `_db` variable (refer to the comments for more
  info)
- `travis.linux` and `travis.osx`: provide a link to the installers
- `appveyor.ps1`: provide a link to the installer, and list the components that
  need to be installed
- `travis.yml` and `appveyor.yml`: add the version to the CI rosters


### GCC version update

Update the `gcc_major_versions` and `gcc_minor_versions` ranges in
`discovery.jl` to cover the new version.
