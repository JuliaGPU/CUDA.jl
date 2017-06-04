# Reflection

Because of using a different compilation toolchain, CUDAnative.jl offers counterpart
functions to the `code_` functionality from Base:

```@docs
CUDAnative.code_llvm
CUDAnative.code_ptx
CUDAnative.code_sass
```


## Convenience macros

For ease of use, CUDAnative.jl also implements `@code_` macros wrapping the above reflection
functionality. These macros determines the type of arguments (taking into account GPU type
conversions), and call the underlying `code_` function. In addition, these functions
understand the `@cuda` invocation syntax, so you conveniently put them in front an existing
`@cuda` invocation.

```@docs
CUDAnative.@code_lowered
CUDAnative.@code_typed
CUDAnative.@code_warntype
CUDAnative.@code_llvm
CUDAnative.@code_ptx
CUDAnative.@code_sass
```
