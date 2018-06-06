CUDAapi.jl
==========

*Reusable components for CUDA API development.*

**Build status**: [![][gitlab-img]][gitlab-url] [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url]

**Code coverage**: [![coverage report](https://gitlab.com/JuliaGPU/CUDAapi.jl/badges/master/coverage.svg)](https://gitlab.com/JuliaGPU/CUDAapi.jl/commits/master)

[gitlab-img]: https://gitlab.com/JuliaGPU/CUDAapi.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CUDAapi.jl/commits/master

[travis-img]: https://travis-ci.org/JuliaGPU/CUDAapi.jl.svg?branch=master
[travis-url]: https://travis-ci.org/JuliaGPU/CUDAapi.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/e41yic5p5ru018mf/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/maleadt/cudaapi-jl


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
