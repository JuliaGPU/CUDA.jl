# CUDA.jl release notes

## In development: CUDA.jl 4.0

### Breaking changes

- CUDNN and CUTENSOR have been split off into separate packages
  ([#1624](https://github.com/JuliaGPU/CUDA.jl/pull/1624)). This should improve load times,
  since most users do not rely on the functionality from these modules.

- Binaries (like the CUDA runtime, CUDNN, etc) are now provided through first-class JLL
  packages ([#1629](https://github.com/JuliaGPU/CUDA.jl/pull/1629)). This makes it possible
  to build JLLs for applications and libraries that rely on the CUDA runtime. As a result,
  the `JULIA_CUDA_USE_BINARYBUILDER` and `JULIA_CUDA_VERSION` environment variables have
  been replaced with Preferences.jl-based settings; refer to the documentation for more
  information.

### New features

...
