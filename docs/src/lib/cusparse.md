# cuSPARSE

```@meta
CurrentModule = cuSPARSE
```

cuSPARSE.jl offers sparse arrays on the GPU, and NVIDIA's sparse linear algebra routines at
three levels:

- The sparse array types, [`CuSparseVector`](@ref), [`CuSparseMatrixCSR`](@ref),
  [`CuSparseMatrixCSC`](@ref), [`CuSparseMatrixCOO`](@ref) and
  [`CuSparseMatrixBSR`](@ref), work with `SparseArrays` and `LinearAlgebra`: convert from
  and to `SparseMatrixCSC` and `CuMatrix`, broadcast, multiply with `*` and `mul!`, add,
  and solve triangular systems with `ldiv!`. This should be the first choice.
- The functions documented below wrap individual cuSPARSE routines, for when more control
  is needed. They are named after the operation they compute, as in BLAS, or after the
  cuSPARSE routine.
- The generated bindings in `libcusparse.jl` follow NVIDIA's C API, and change with the CUDA
  toolkit. Use them together with [`handle`](@ref) and the descriptors below.

The wrappers follow `LinearAlgebra.BLAS`'s conventions. Character arguments select the
variant to compute: `transa`, `transb` and `transx` are `'N'`, `'T'` or `'C'`, `uplo` is
`'U'` or `'L'`, and `diag` is `'U'` for a unit diagonal or `'N'`. Functions whose name ends
in `!` overwrite one of their arguments, the result, and return it. The index base of the sparse operands is the keyword
`index`: `'O'` (the default) for one-based indices as in Julia, or `'Z'` for zero-based
ones. Routines that offer several algorithms take a cuSPARSE algorithm enum, such as
`CUSPARSE_SPMV_CSR_ALG2`, as an optional last positional argument.

Most wrappers use cuSPARSE's generic API, and check that the installed cuSPARSE supports
them. Some only exist in NVIDIA's legacy API: the BSR triangular solves [`sv2!`](@ref) and
[`sm2!`](@ref), the incomplete factorizations [`ic02!`](@ref) and [`ilu02!`](@ref),
[`gemvi!`](@ref), [`color`](@ref) and [`prune`](@ref). NVIDIA has deprecated these, as well
as the generic [`axpby!`](@ref), [`rot!`](@ref) and [`vv!`](@ref), so they may disappear
from a future CUDA toolkit.

Several wrappers share their name with a function in `LinearAlgebra`, e.g.
`cuSPARSE.axpby!` and `LinearAlgebra.axpby!`, but are different functions with different
arguments. Qualify them with the module name.

## Public

```@autodocs
Modules = [cuSPARSE]
Private = false
```

## Private

```@autodocs
Modules = [cuSPARSE]
Public = false
```
