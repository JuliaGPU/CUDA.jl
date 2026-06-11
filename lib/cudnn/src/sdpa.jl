@doc raw"""
    cudnnSDPAForward(q, k, v; o...)
    cudnnSDPAForward!(out, q, k, v; o...)

Fused scaled-dot-product ("flash") attention forward pass, built on cuDNN's modern backend
graph API using the dedicated fused-attention operation
(`CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`). This is the recommended path for transformer
attention and supersedes the legacy [`cudnnMultiHeadAttnForward`](@ref) API.

Computes, per batch and head,

```math
\mathrm{out} = \mathrm{softmax}\!\left(\mathrm{scale}\cdot Q K^\top\right) V
```

as a single fused (flash-attention) kernel: the `sq × skv` score matrix is never written to
global memory.

# Layout

`q`, `k`, `v` and `out` are 4-D `CuArray`s in **(head_dim, num_heads, seq_len, batch)** order
(head dimension fastest / innermost, as cuDNN requires). This matches NNlib's internal
multi-head attention layout, so no permute is needed to interoperate with it. With head dim `d`:
* `q`  :  `(d, h, sq, b)`
* `k`  :  `(d, h, skv, b)`
* `v`  :  `(d, h, skv, b)`
* `out`:  `(d, h, sq, b)`  (allocated by the non-`!` form)

# Keyword arguments

* `scale::Real = 1/√d`: scalar multiplied into the `QKᵀ` scores before softmax.

# Notes

Forward inference only, in `Float16`/`BFloat16`, with equal head counts for `q`/`k`/`v` (no
GQA/MQA), no masking, no dropout, and no training/backward. The fused engine requires an
Ampere (sm_80) or newer GPU. `Float32`/`Float64` are **not** supported by cuDNN's fused
attention engine.
"""
cudnnSDPAForward, cudnnSDPAForward!


struct SDPAPlan
    plan::cudnnBackendDescriptor
    workspace_size::Int
    uids::Vector{Int64}
    keepalive::Vector{Any}   # descriptors that must outlive the plan
end

# fixed tensor UIDs in the graph
const SDPA_UID_Q = 1
const SDPA_UID_K = 2
const SDPA_UID_V = 3
const SDPA_UID_O = 4
const SDPA_UID_SCALE = 5

const SDPA_PLAN_CACHE = Dict{Any,SDPAPlan}()
const SDPA_PLAN_CACHE_LOCK = ReentrantLock()

# A Julia (head_dim, nheads, seq_len, batch) column-major array (NNlib's internal attention
# layout, a BSHD physical layout) maps to the cuDNN logical tensor [b, h, s, d] with head_dim
# innermost (stride 1). The dedicated SDPA op takes batch and heads as native leading dims.
_attn_dims(d, s, h, b) = Int64[b, h, s, d]
_attn_strides(d, s, h, b) = Int64[d*h*s, d, d*h, 1]

function _build_sdpa_plan(T::DataType, d, sq, skv, h, b)
    dt = cudnnDataType(T)
    F = CUDNN_DATA_FLOAT
    keep = Any[]
    tensor(; kw...) = (t = backend_tensor(; kw...); push!(keep, t); t)

    Q = tensor(uid=SDPA_UID_Q, dims=_attn_dims(d,sq,h,b),  strides=_attn_strides(d,sq,h,b),  dtype=dt)
    K = tensor(uid=SDPA_UID_K, dims=_attn_dims(d,skv,h,b), strides=_attn_strides(d,skv,h,b), dtype=dt)
    V = tensor(uid=SDPA_UID_V, dims=_attn_dims(d,skv,h,b), strides=_attn_strides(d,skv,h,b), dtype=dt)
    O = tensor(uid=SDPA_UID_O, dims=_attn_dims(d,sq,h,b),  strides=_attn_strides(d,sq,h,b),  dtype=dt)
    scale = tensor(uid=SDPA_UID_SCALE, dims=Int64[1,1,1,1], strides=Int64[1,1,1,1],
                   dtype=F, by_value=true)

    op = cudnnBackendDescriptor(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
    setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, Q)
    setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, K)
    setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, V)
    setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, O)
    setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, scale)
    bfinalize!(op)

    graph = operation_graph(cudnnBackendDescriptor[op])
    push!(keep, op, graph)
    heur, cfgs = engine_configs(graph)
    push!(keep, heur)

    for cfg in cfgs
        plan = try_execution_plan(cfg)
        plan === nothing && continue
        push!(keep, cfg)
        uids = Int64[SDPA_UID_Q, SDPA_UID_K, SDPA_UID_V, SDPA_UID_O, SDPA_UID_SCALE]
        return SDPAPlan(plan, Int(plan_workspace_size(plan)), uids, keep)
    end
    error("cuDNN: no supported fused-attention engine for T=$T d=$d sq=$sq skv=$skv h=$h b=$b")
end

function _sdpa_plan(T, d, sq, skv, h, b)
    key = (T, d, sq, skv, h, b)
    lock(SDPA_PLAN_CACHE_LOCK) do
        get!(() -> _build_sdpa_plan(T, d, sq, skv, h, b), SDPA_PLAN_CACHE, key)
    end
end


function cudnnSDPAForward(q, k, v; o...)
    out = similar(q, size(q))
    cudnnSDPAForward!(out, q, k, v; o...)
end

function cudnnSDPAForward!(out, q, k, v; scale::Real = 1/sqrt(size(q,1)), causal::Bool = false)
    T = eltype(q)
    @assert T in (Float16, BFloat16) "cudnnSDPAForward supports Float16/BFloat16, got $T (cuDNN's fused attention does not support Float32/Float64)"
    @assert eltype(k) == eltype(v) == eltype(out) == T "q, k, v, out must share element type"
    @assert ndims(q) == ndims(k) == ndims(v) == ndims(out) == 4 "q, k, v, out must be 4-D (d, h, s, b)"
    # Causal masking is not yet supported. On cuDNN ≤ 9.20 there is no usable path through the
    # raw backend API: the SDPA node's clean score-modifier subgraph
    # (CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH) requires cuDNN ≥ 9.21 (no CUDNN_jll yet), the
    # block-mask attribute is block-sparse rather than element-wise, and the primitive
    # matmul→softmax→matmul graph yields no fused engine from raw backend calls. Revisit and
    # implement via the score-modifier subgraph once CUDNN_jll ≥ 9.21 ships.
    causal && throw(ArgumentError("causal masking is not yet supported by cudnnSDPAForward"))
    d, h, sq, b = size(q)
    dk, hk, skv, bk = size(k)
    @assert (dk, hk, bk) == (d, h, b) "k must be (d, h, skv, b) matching q's d, h, b"
    @assert size(v) == (d, h, skv, b) "v must be (d, h, skv, b)"
    @assert size(out) == (d, h, sq, b) "out must be (d, h, sq, b)"
    @assert d % 8 == 0 "head_dim must be a multiple of 8, got $d"

    plan = _sdpa_plan(T, d, sq, skv, h, b)

    workspace = cudnnTempSpace(plan.workspace_size)
    wsptr = workspace === nothing ? C_NULL : pointer(workspace)
    scalebuf = Float32[scale]
    pointers = Any[pointer(q), pointer(k), pointer(v), pointer(out), pointer(scalebuf)]

    vp = variant_pack(uids=plan.uids, pointers=pointers, workspace=wsptr)
    GC.@preserve q k v out scalebuf workspace vp begin
        cudnnBackendExecute(handle(), plan.plan.ptr, vp.ptr)
    end
    return out
end


@doc raw"""
    cudnnSDPABackward(dout, q, k, v, out; scale, causal)

Backward pass for [`cudnnSDPAForward`](@ref), returning the gradients `(dq, dk, dv)` with
respect to `q`, `k`, `v` given the output gradient `dout` and the forward output `out`.

**Not yet implemented — placeholder.** cuDNN's fused-attention backward is not reachable on
cuDNN ≤ 9.20 through the backend graph API:

* The dedicated `CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR` (type 43) does not finalize into a
  valid operation graph in this version (`CUDNN_STATUS_BAD_PARAM` with Q/K/V/O/Stats/scale/dO →
  dQ/dK/dV all set); it is effectively undocumented/incomplete before cuDNN 9.21.
* The documented path — a score-modifier subgraph on the SDPA node
  (`CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH`), with the forward additionally emitting its FP32
  log-sum-exp `stats` (`CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC`, already verified to work) —
  requires cuDNN ≥ 9.21, for which no `CUDNN_jll` artifact exists yet.

This is the same cuDNN-version limitation that blocks `causal=true` in
[`cudnnSDPAForward`](@ref). Once `CUDNN_jll ≥ 9.21` is available, implement this by having the
forward output the `stats` tensor and building the backward via the SDPA score-modifier
subgraph mechanism.
"""
function cudnnSDPABackward(dout, q, k, v, out; o...)
    error("cudnnSDPABackward is not yet implemented: cuDNN's fused-attention backward requires " *
          "cuDNN ≥ 9.21 (no CUDNN_jll available yet). See the cudnnSDPABackward docstring.")
end
