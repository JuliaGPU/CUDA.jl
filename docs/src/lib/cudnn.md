# cuDNN

```@meta
CurrentModule = cuDNN
```

cuDNN.jl wraps NVIDIA cuDNN for use with `CuArray` values. New wrappers are built around
cuDNN's backend and graph APIs: describe the operation graph in Julia, let cuDNN choose an
execution engine, cache the resulting plan, and execute it against concrete arrays.

The older fixed-function wrappers are still available for compatibility. They are kept
separate from the modern wrappers so downstream packages can migrate without losing access
to legacy entry points.

## Wrapper Design

The package has four layers:

- Generated bindings in `libcudnn.jl` mirror the C API and expose the raw cuDNN constants,
  structs, and functions. These bindings are not deprecated by cuDNN.jl.
- Backend descriptor helpers wrap `cudnnBackendDescriptor_t` for the generic cuDNN backend
  API. Descriptors are indexed with short field symbols derived from the attribute enums —
  `d[:qdesc] = tensor` on an SDPA node sets `CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC`, and
  `plan[:workspace_size, Int64]` reads an attribute back — with typed `setattr!`/`getattr`
  underneath for cases the derived names don't cover.
- The graph frontend uses `Graph` and `Tensor` objects to describe tensors, scalar values,
  virtual intermediates, and operations before lowering them to backend descriptors.
- Operation wrappers such as `attention!`, `convolution!`, pooling, and batch normalization
  provide the Julia-facing API used by downstream packages such as NNlib and LuxLib.

The modern wrappers should prefer Julia names over C names. In-place methods take the output
array first and end in `!`. Allocating methods, when present, call the in-place method after
allocating the output. Keyword arguments should use Julia values such as symbols, integers,
booleans, and `nothing`; conversion to cuDNN enums and descriptors belongs near the cuDNN
call.

Shape and stride handling should follow Julia array order at the public API boundary. When
cuDNN expects a different backend order, the graph tensor records that order explicitly and
the lowering step translates dimensions and strides. This keeps public calls consistent with
the arrays users pass in while still matching cuDNN's graph requirements.

The generated binding layer still uses cuDNN names because it mirrors the C headers. The
legacy compatibility layer also keeps historical cuDNN-style names, such as
`cudnnConvolutionForward!`, so existing callers keep working during the transition.

## Graph Execution

A graph is built in three steps:

1. Create a `Graph` with the desired IO, intermediate, and compute data types.
2. Add `Tensor` objects for inputs, outputs, scalars, and virtual intermediates.
3. Add operations, call `build!`, and then call `execute!` with concrete array and scalar
   bindings.

`build!` validates the graph, assigns tensor ids, lowers operations to backend descriptors,
asks cuDNN heuristics for candidate engines, and finalizes the first supported execution
plan; engine selection honors the `deterministic`, `math_mode`, and `max_workspace`
keywords. `is_supported(g)` runs the same pipeline but returns `false` instead of throwing
when no engine finalizes. `execute!` builds a variant pack from the supplied bindings and
runs the cached plan with an automatically allocated workspace.

High-level wrappers should normally cache built graphs through `cached_graph`, with a key
that includes every layout, type, alignment, and option that affects engine selection.
`UnsupportedGraphError` is the expected signal that cuDNN could not build a plan for the
requested graph, and `cached_graph` caches that outcome so repeated calls stay cheap.
Callers with an equivalent generic implementation should test errors with
`graph_unsupported` and fall back when it returns `true`.

Op factories can create by-value tensors implicitly, such as the scale scalar of an SDPA
node or the fill value of its causal mask. These must be bound at execution time like any
other non-virtual tensor; `tensor(g, name)` looks them up by name.

The graph layer is intentionally close to NVIDIA's cudnn-frontend model, but expressed as
ordinary Julia objects and keyword-heavy factory functions instead of C++ builders. The
`Tensor` objects describe metadata only; concrete device pointers are supplied later when
calling `execute!`.

Execution plans are cached per cuDNN handle. cuDNN execution plans are handle-bound, so the
cache lives with the pooled handle and is destroyed before the handle itself. Cache keys must
include the input and output layouts, element types, pointer alignments, operation options,
math mode, determinism, and workspace limit.

## Operation Wrappers

The ops layer is the preferred public surface for downstream packages. These methods accept
preallocated outputs and hide backend descriptors for common calls:

- `attention!` and `attention_backward!` use unified SDPA graph operations for
  `(head_dim, heads, sequence, batch)` tensors. Forward supports optional saved stats and
  top-left causal masking, and backward accepts the same causal mode. Forward and backward
  support dense padding masks through Int32 `seq_len_q` and `seq_len_kv` tensors shaped
  `(1, 1, 1, batch)`. Dropout and bias are not wired yet. `attention_supported` and
  `attention_backward_supported` report whether the fused path can run a given call, so
  callers can pick a fallback up front; engine coverage differs between forward and
  backward on some architectures.
- `convolution!`, `convolution_data_gradient!`, and `convolution_filter_gradient!` use graph
  convolution operations for plain convolutions and gradients, with native asymmetric
  padding. Fused bias, residual, and activation cases try graph fusion and fall back to
  plain convolution plus broadcast work when cuDNN has no supported fused plan; asymmetric
  padding without engine support falls back to padding the input manually.
- `maxpool!`, `∇maxpool!`, `meanpool!`, and `∇meanpool!` use graph resample operations
  when cuDNN can build a plan.
- `batchnorm_training!`, `batchnorm_inference!`, and `batchnorm_gradient!` use graph norm
  operations when cuDNN supports the requested layout and fall back to fixed-function batch
  normalization for unsupported graph plans or non-default scaling.

The wrappers should return early for empty arrays before touching cuDNN. Size-one dimensions
should use contiguous-consistent strides because cuDNN plan selection is sensitive to tensor
metadata. Pointer alignment is part of both the tensor descriptor and the graph cache key.

## Descriptor Rules

Fixed-function cuDNN calls take tensor and operator descriptors directly. That pattern is
fine for low-level use, but new ops should avoid making descriptors part of ordinary call
signatures: descriptor construction is cheap enough to keep close to the cuDNN call, built
from ordinary Julia values, converting `nothing` to `C_NULL` or `CU_NULL` only at the
low-level call boundary.

Workspaces should be requested as close as possible to the cuDNN call or execution plan.
cuDNN can make workspace requirements depend on details beyond the visible input shapes, so
workspace allocation should not be moved far away from the final algorithm or plan choice.

cuDNN distinguishes training and inference inconsistently across operation families. The ops
layer should expose that distinction explicitly when it matters, as batch normalization does,
and avoid inventing a training flag for operations where cuDNN has no such mode.

## Legacy Wrappers

The cordoned legacy wrappers preserve the historical imperative cuDNN API: activation,
pooling, convolution, normalization, multi-head attention, elementwise ops, reductions,
and related tests. Source files live in `lib/cudnn/src/legacy`, and matching tests live in
`lib/cudnn/test/legacy`. Nothing outside that directory depends on the wrappers in it, so
the whole directory can be deleted in the next breaking release once downstream packages
require the graph-backed ops layer.

Legacy wrappers should remain available until the next breaking release, but new APIs should
not grow around them. New functionality should target the backend and graph layers when
cuDNN exposes the needed operation there, or use a small compatibility wrapper that can be
removed with the rest of the legacy surface.

Softmax, dropout, and RNN wrappers are fixed-function survivors rather than legacy code:
cuDNN has no graph replacement for dropout and RNN, and NNlib's softmax hook still uses
the fixed-function entry points. Descriptor plumbing shared with the surviving layers
(array-based tensor and filter descriptor constructors, convolution algorithm selection)
also lives outside `legacy`. The generated C bindings remain available regardless of the
legacy-wrapper deprecation schedule.

## Debugging

cuDNN can emit detailed diagnostics for graph support and plan selection. Set
`CUDNN_LOGLEVEL_DBG=3` in the environment, or start Julia with `JULIA_DEBUG=cuDNN` (or
`-g2`) so the cuDNN log callback registered at initialization reports messages through
Julia logging.

When graph construction fails with `UnsupportedGraphError`, check the tensor dimensions,
strides, data types, pointer alignments, math mode, determinism setting, and workspace
limit. Unsupported graph plans are normal for some layouts and operation combinations; the
ops layer should fall back only when the fallback has the same semantics.

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
