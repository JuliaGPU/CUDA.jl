CUDAapi.jl
==========

*Reusable components for CUDA API development.*

**Build status**: [![][buildbot-julia06-img]][buildbot-julia06-url] [![][buildbot-juliadev-img]][buildbot-juliadev-url] [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url]

[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAapi-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAapi-julia06-x86-64bit
[buildbot-juliadev-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAapi-juliadev-x86-64bit&name=julia%20dev
[buildbot-juliadev-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAapi-juliadev-x86-64bit

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
