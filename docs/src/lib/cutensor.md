# cuTENSOR

```@meta
CurrentModule = cuTENSOR
```

cuTENSOR.jl offers NVIDIA's tensor primitives at three levels:

- [`CuTensor`](@ref) pairs a `CuArray` with a label for each of its dimensions. Tensors
  can be added, subtracted and multiplied with `+`, `-`, `*`, `axpy!`, `axpby!` and
  `mul!`, which contracts the dimensions they share.
- The functions documented below wrap cuTENSOR's operations, for when more control is
  needed: [`contract!`](@ref), [`cuTENSOR.permute!`](@ref), [`reduce!`](@ref),
  [`elementwise_binary!`](@ref) and [`elementwise_trinary!`](@ref).
- The generated bindings in `libcutensor.jl` follow NVIDIA's C API, and change with the
  cuTENSOR version. Use them together with [`cuTENSOR.handle`](@ref),
  [`CuTensorDescriptor`](@ref) and [`CuTensorPlan`](@ref).

The operations take plain arrays and their modes, a `Char` or integer that labels each
dimension, as in Einstein notation. Each array comes with a scaling factor and a unary
operator from cuTENSOR's `cutensorOperator_t` (`OP_IDENTITY`, `OP_CONJ`, ...), and the
result is written in place:

```julia
using CUDA, cuTENSOR
using cuTENSOR: contract!, OP_IDENTITY

A = CUDA.rand(Float32, 4, 5)
B = CUDA.rand(Float32, 5, 6)
C = CUDA.zeros(Float32, 4, 6)

# C[i,k] = sum over j of A[i,j] * B[j,k]
contract!(1, A, ['i', 'j'], OP_IDENTITY, B, ['j', 'k'], OP_IDENTITY,
          0, C, ['i', 'k'], OP_IDENTITY, OP_IDENTITY)
```

Every operation is planned before it is executed. To execute one several times, make
a [`CuTensorPlan`](@ref) with the corresponding `plan_*` function and pass it to the
operation, either as the `plan` keyword argument or as its first argument. Keyword
arguments select the algorithm, the workspace size, the precision of the computation
([`cutensorComputeDescriptorEnum`](@ref)) and whether to compile a kernel for the operation;
the enum values they take, like `ALGO_DEFAULT` and `WORKSPACE_MAX`, are defined without
NVIDIA's `CUTENSOR_` prefix.

The names of the operations are not exported: call them qualified, or import them.
`cuTENSOR.permute!` in particular is a different function than `Base.permute!`.

## Public

```@autodocs
Modules = [cuTENSOR]
Private = false
```

## Private

```@autodocs
Modules = [cuTENSOR]
Public = false
```
