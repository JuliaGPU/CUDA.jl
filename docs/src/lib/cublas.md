# cuBLAS

```@meta
CurrentModule = cuBLAS
```

cuBLAS.jl offers NVIDIA's dense linear algebra library at three levels:

- `LinearAlgebra` functions like `*`, `mul!`, `dot`, `norm`, `axpy!` and `ldiv!` with
  triangular matrices work on `CuArray`s, and should be the first choice.
- The functions documented below wrap individual cuBLAS routines, for when more control is
  needed, or for operations that `LinearAlgebra` does not have, like batched products and
  factorizations. They are named after the BLAS routine they compute; the routines of
  cuBLASXt, which distribute large products over several GPUs and accept host memory, have
  an `xt_` prefix.
- The generated bindings in `libcublas.jl` and `libcublasLt.jl` follow NVIDIA's C API, and
  change with the CUDA toolkit. Use those in `libcublas.jl` together with
  [`handle`](@ref); cuBLASLt needs its own handle, created with `cublasLtCreate`.

The wrappers follow `LinearAlgebra.BLAS`'s conventions. Character arguments select the
variant to compute (`trans`, `uplo`, `side`, `diag`), functions ending in `!` overwrite
their output argument, and the functions without `!` allocate it. Unlike
`LinearAlgebra.BLAS`, the level-1 functions take the number of elements first and read
every array with its own stride, and some can write their result to a device-side scalar
to avoid synchronizing. cuBLAS chooses between its 32-bit and 64-bit entry points
automatically.

## Public

```@autodocs
Modules = [cuBLAS]
Private = false
```

## Private

```@autodocs
Modules = [cuBLAS]
Public = false
```
