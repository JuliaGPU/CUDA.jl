# cuDNN

```@meta
CurrentModule = cuDNN
```

cuDNN.jl provides generated C bindings, a graph frontend, and graph-backed operations for
`CuArray` values.

## Structure

- `libcudnn.jl` contains the generated C API.
- `backend.jl` manages generic backend descriptors.
- `graph/` defines `Graph`, `Tensor`, operation factories, plan selection, and execution.
- `ops/` defines the Julia-facing attention, convolution, pooling, and batch-normalization
  APIs.
- `legacy/` contains fixed-function compatibility wrappers.

Public operations use Julia array order: spatial dimensions, channels, then batch. Graph
tensors record any ordering required by cuDNN.

## Graphs

Create a `Graph`, add tensors and operations, then call `build!` and `execute!`. `Tensor`
objects contain metadata; `execute!` binds them to arrays or scalar values.

`build!` validates and lowers the graph, queries cuDNN heuristics, and selects an execution
plan. `deterministic`, `math_mode`, and `max_workspace` constrain plan selection.
`is_supported` performs the same build and returns a boolean.

`cached_graph` caches plans and unsupported configurations per cuDNN handle.
`graph_unsupported` identifies errors suitable for an equivalent fallback.

## Operations

- `attention!` and `attention_backward!` implement fused SDPA for
  `(head_dim, heads, sequence, batch)` tensors, including causal and sequence-length masks.
- `convolution!` and its gradient methods support grouped and asymmetric convolutions,
  optional bias and residual inputs, and pointwise activation.
- `maxpool!`, `meanpool!`, `∇maxpool!`, and `∇meanpool!` use graph resampling.
- `batchnorm_training!`, `batchnorm_inference!`, and `batchnorm_gradient!` implement spatial
  batch normalization.

Support predicates build and cache a plan without executing it.

## Compatibility

Fixed-function compatibility wrappers and tests live under `src/legacy` and `test/legacy`.
The graph and operation layers do not depend on them. Softmax, dropout, and RNN remain
outside `legacy`.

## Debugging

Set `CUDNN_LOGLEVEL_DBG=3` or start Julia with `JULIA_DEBUG=cuDNN` to report cuDNN
diagnostics through Julia logging.

## Public

```@autodocs
Modules = [cuDNN]
Private = false
```

## Private

```@autodocs
Modules = [cuDNN]
Public = false
```
