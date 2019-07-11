CUDAapi.jl
==========

*Reusable components for CUDA API development.*

**Code coverage**: [![][codecov-img]][codecov-url]

[codecov-img]: https://codecov.io/gh/JuliaGPU/CUDAapi.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CUDAapi.jl


This package provides some reusable functionality for programming against CUDA or NVIDIA
APIs. It is meant to be used by package developers, and is not expected to export
functionality useful for application programmers.



Usage
-----

### Compatibility

The file `src/compatibility.jl` contains hard-coded databases with software and hardware
compatibility information that cannot be queried from APIs.


### Discovery

The file `src/discovery.jl` defines helper methods for discovering the NVIDIA driver and
CUDA toolkit, as well as some more generic methods to find libraries and binaries relative
to eg. the location of the driver or toolkit.



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
