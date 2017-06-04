# libdevice

CUDAnative.jl provides wrapper functions for the mathematical routines in `libdevice`,
CUDA's device math library. Many of these functions implement an interface familiar to
similar functions in `Base`, but it is currently impossible to transparently dispatch to
these device functions. As a consequence, users should prefix calls to math functions (eg.
`sin` or `pow`) with the CUDAnative module name.

WIP
