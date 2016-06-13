CUDAdrv.jl
==========

CUDA driver programming interface for Julia, including auxiliary utilities for other
CUDA-related packages (most notably CUDArt.jl, CUDAnative.jl).


Debugging
---------

For extra information about what's happening behind the scenes, you can enable extra output
by defining the `DEBUG` environment variable, or `TRACE` for even more information. These
features are incompatible with precompilation (the debugging code is enabled statically in
order to avoid any run-time overhead), so you need to wipe the compile cache or run julia
using `--compilecache=no` (enabling colors with `--color=yes` also prettifies the output).
