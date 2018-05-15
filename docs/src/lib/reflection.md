# Reflection

Because of using a different compilation toolchain, CUDAnative.jl offers counterpart
functions to the `code_` functionality from Base:

```@docs
CUDAnative.code_llvm
CUDAnative.code_ptx
CUDAnative.code_sass
```


## Convenience macros

For ease of use, CUDAnative.jl also implements `@device_code_` macros wrapping
the above reflection functionality. These macros evaluate the expression
argument, while tracing compilation and finally printing or returning the code
for every invoked CUDA kernel. Do note that this evaluation can have side
effects, as opposed to similarly-named `@code_` macros in Base which are free of
side effects.

```@docs
CUDAnative.@device_code_lowered
CUDAnative.@device_code_typed
CUDAnative.@device_code_warntype
CUDAnative.@device_code_llvm
CUDAnative.@device_code_ptx
CUDAnative.@device_code_sass
CUDAnative.@device_code
```
