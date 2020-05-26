# Compiler

## Execution

The main entry-point to the compiler is the `@cuda` macro:

```@docs
@cuda
```

If needed, you can use a lower-level API that lets you inspect the compiler kernel:

```@docs
cudaconvert
cufunction
CUDA.HostKernel
CUDA.version
CUDA.maxthreads
CUDA.registers
CUDA.memory
```


## Reflection

If you want to inspect generated code, you can use macros that resemble functionality from
the InteractiveUtils standard library:

```
@device_code_lowered
@device_code_typed
@device_code_warntype
@device_code_llvm
@device_code_ptx
@device_code_sass
@device_code
```

These macros are also available in function-form:

```
CUDA.code_typed
CUDA.code_warntype
CUDA.code_llvm
CUDA.code_ptx
CUDA.code_sass
```

For more information, please consult the GPUCompiler.jl documentation. Only the `code_sass`
functionality is actually defined in CUDA.jl:

```@docs
@device_code_sass
CUDA.code_sass
```
