# cuTensorNet

```@meta
CurrentModule = cuTensorNet
```

cuTensorNet.jl wraps NVIDIA's cuTensorNet library, part of cuQuantum, for contracting and
decomposing tensor networks and simulating quantum circuits. It has no integration with
Base or LinearAlgebra functions; it offers two levels:

- The functions and types documented below wrap cuTensorNet's routines and objects. They
  fall into three groups:
  - **Network contractions**: build a [`CuNetwork`](@ref) from input tensors, find a
    contraction path with [`rehearse_contraction`](@ref), and contract it with
    [`contract!`](@ref), optionally computing gradients with
    [`compute_gradients!`](@ref).
  - **Tensor decompositions**: [`tensor_qr!`](@ref), [`tensor_svd!`](@ref) and
    [`gate_split!`](@ref), on `CuArray`s with mode labels or on cuTENSOR's `CuTensor`s.
  - **States**: build a [`CuNetworkState`](@ref) by applying operators to it, then compute
    the state, its amplitudes, expectation values, marginals or samples, or project it on
    an MPS.
- The generated bindings in `libcutensornet.jl` follow NVIDIA's C API, and change with the
  cuQuantum release. Use them together with [`handle`](@ref).

The wrappers use snake-case names, and share these conventions:

- Modes are labeled with integers (or, for the decompositions, characters). Modes with the
  same label are contracted. The modes of a state are numbered from 0, as in NVIDIA's API.
- Objects keep the arrays they are given alive, and the library uses their memory directly:
  change their contents only where a function says that is supported.
- Settings that NVIDIA exposes as attributes are passed as keyword structs
  ([`OptimizerConfig`](@ref), [`SVDConfig`](@ref),
  [`NetworkAutotunePreferences`](@ref)), or with [`attribute!`](@ref) and
  [`configure!`](@ref), which take NVIDIA's attribute enums and values of the type NVIDIA
  documents for them. The same goes for the enum-typed keywords of the wrappers, such as
  `algo` of [`gate_split!`](@ref).
- Functions that launch work take a `stream` keyword, which defaults to the task's stream.
  The computations on states wait for their result; the others don't.

`CuTensorNetwork`, `perform_contraction!`, `AutoTune`, `NoAutoTune` and
`AutotunePreferences` wrap NVIDIA's deprecated network descriptors and contraction plans,
and are deprecated in favor of [`CuNetwork`](@ref).

## Public

```@autodocs
Modules = [cuTensorNet]
Private = false
```

## Private

```@autodocs
Modules = [cuTensorNet]
Public = false
```
