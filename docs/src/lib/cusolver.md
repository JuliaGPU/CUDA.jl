# cuSOLVER

```@meta
CurrentModule = cuSOLVER
```

cuSOLVER.jl offers NVIDIA's dense and sparse solvers at three levels:

- `LinearAlgebra` functions like `\`, `lu`, `qr`, `cholesky`, `eigen` and `svd` work on
  `CuArray`s, and should be the first choice. The algorithm used by `svd` can be chosen
  with the types below.
- The functions documented below wrap individual cuSOLVER routines, for when more control
  is needed. They are named after the LAPACK routine they compute, and many also extend
  the corresponding function in `LinearAlgebra.LAPACK`.
- The generated bindings in `libcusolver.jl` follow NVIDIA's C API, and change with the
  CUDA toolkit. Use them together with [`dense_handle`](@ref) and
  [`CuSolverParameters`](@ref).

The wrappers follow `LinearAlgebra.LAPACK`'s conventions. Character arguments select the
variant to compute (`uplo`, `trans`, `jobz`, ...), and arguments are overwritten in place.
Errors in the arguments are thrown, while the `info` value of a factorization is returned,
so that you can decide what a singular or indefinite matrix means.

cuSOLVER offers many routines twice, in its legacy API and its 64-bit generic API. The
wrappers use the generic API when the installed cuSOLVER provides it, and the legacy API
otherwise. [`getrf!`](@ref) and [`getrs!`](@ref) choose by the type of the pivot vector
instead: 32-bit pivots use the legacy API, 64-bit pivots the generic one.

## Public

```@autodocs
Modules = [cuSOLVER]
Private = false
```

## Private

```@autodocs
Modules = [cuSOLVER]
Public = false
```
