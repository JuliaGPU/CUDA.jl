# CUDAdrv.jl

**CUDA driver programming interface for Julia.**

[![Coverage Status](https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUDAdrv.jl)


Debugging
---------

For extra information about what's happening behind the scenes, you can enable extra output
by defining the `DEBUG` environment variable, or `TRACE` for even more information. These
features are incompatible with precompilation (the debugging code is enabled statically in
order to avoid any run-time overhead), so you need to wipe the compile cache or run julia
using `--compilecache=no` (enabling colors with `--color=yes` also prettifies the output).
