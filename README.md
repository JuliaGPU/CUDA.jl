CUDAapi.jl
==========

*Reusable components for CUDA API development.*

**Build status**: [![][buildbot-julia05-img]][buildbot-julia05-url] [![][buildbot-julia06-img]][buildbot-julia06-url] [![][buildbot-juliadev-img]][buildbot-juliadev-url]

[buildbot-julia05-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAapi-julia05-x86-64bit&name=julia%200.5
[buildbot-julia05-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAapi-julia05-x86-64bit
[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAapi-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAapi-julia06-x86-64bit
[buildbot-juliadev-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAapi-juliadev-x86-64bit&name=julia%20dev
[buildbot-juliadev-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAapi-juliadev-x86-64bit


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


## Logging

The file `src/logging.jl` contains some methods for fine-grained logging at different
log-levels. It is excepted to be deprecated and removed when
[MicroLogging.jl](https://github.com/c42f/MicroLogging.jl) is usable.
