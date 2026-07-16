# Compiler

```@meta
CurrentModule = CUDACore
```

## Execution

The main entry-point to the compiler is the `@cuda` macro:

```@docs
@cuda
```

If needed, you can use a lower-level API that lets you inspect the compiled kernel:

```@docs
KernelInvocation
kernel_launch
rebind
cudaconvert
cufunction
AbstractKernel
HostKernel
version
maxthreads
registers
memory
```

Launch a compiled invocation with `kernel_launch(kernel, invocation; launch_kwargs...)`. Arguments
can be rebound immutably with `rebind(invocation, index => value)` before launching.

The PTX compilation target is identified by an `SMVersion`, constructed via the
`sm"..."` string macro:

```@docs
SMVersion
@sm_str
```

To plug in alternative compiler back-ends (e.g. cuTile.jl), `@cuda` dispatches
through a small protocol:

```@docs
AbstractBackend
LLVMBackend
DefaultBackend
kernel_convert
kernel_compile
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

```@meta
CurrentModule = CUDATools
```

```@docs
@device_code_sass
code_sass
```
