# cuStateVec

```@meta
CurrentModule = cuStateVec
```

cuStateVec.jl wraps NVIDIA's cuStateVec library, part of cuQuantum, for simulating quantum
circuits on state vectors. It has no integration with Base or LinearAlgebra functions; it
offers two levels:

- The functions and types documented below wrap cuStateVec's routines on a
  [`CuStateVec`](@ref): applying gates, measuring, sampling, computing expectation values,
  accessing and reordering the elements, and the helper objects for distributed state
  vectors.
- The generated bindings in `libcustatevec.jl` follow NVIDIA's C API, and change with the
  cuQuantum release. Use them together with [`handle`](@ref).

The wrappers are named after the cuStateVec routine they call, in NVIDIA's camel case,
without the `custatevec` prefix and with a `!` when they modify their first argument:
`custatevecApplyMatrix` is [`applyMatrix!`](@ref), and `custatevecComputeExpectation`
[`computeExpectation`](@ref). Routines of a helper object also drop the object's name,
which the type of their first argument gives, and getters drop `Get`:
`custatevecSamplerPreprocess` is [`preprocess!`](@ref), `custatevecSamplerGetSquaredNorm`
[`squaredNorm`](@ref), and `custatevecInitializeStateVector` [`initialize!`](@ref). The
exceptions are [`accessorGet`](@ref) and [`accessorSet!`](@ref), which keep the object's
name rather than shadowing `get` and `set!`, and [`swapParameters`](@ref), which names the
swap parameters that `custatevecDistIndexBitSwapSchedulerGetParameters` returns.

The wrappers share these conventions:

- Qubits, or index bits, are numbered from 0: bit `i` of the index of a state vector's
  amplitudes is qubit `i`. Bit strings returned as integers have the first of the requested
  bits as their least significant bit.
- Matrices are column-major, as Julia stores them, and can be in host or device memory.
- Batched wrappers take one `CuStateVec` that holds `n_svs` state vectors of equal length
  one after the other.
- Measurements take random numbers in `[0, 1)` from the caller, so that they can be
  reproduced. Flags such as `adjoint`, and the enums of the C API (`custatevecCollapseOp_t`,
  `custatevecMatrixMapType_t`, ...), are positional arguments.
- The wrappers run on the task's stream. The ones that return host values, such as
  [`computeExpectation`](@ref) and [`abs2SumArray`](@ref), wait for them.

## Public

```@autodocs
Modules = [cuStateVec]
Private = false
```

## Private

```@autodocs
Modules = [cuStateVec]
Public = false
```
