@doc raw"""
    cudnnSDPAForward(q, k, v; o...)
    cudnnSDPAForward!(out, q, k, v; o...)

Fused scaled-dot-product ("flash") attention forward pass, built on cuDNN's modern backend
graph API using the dedicated fused-attention operation
(`CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`). This is the recommended path for transformer
attention and supersedes the legacy [`cudnnMultiHeadAttnForward`](@ref) API.

Computes, per batch and head,

```math
\mathrm{out} = \mathrm{softmax}\!\left(\mathrm{scale}\cdot Q K'\right) V
```

as a single fused (flash-attention) kernel: the `sq x skv` score matrix is never written to
global memory.

# Layout

`q`, `k`, `v` and `out` are 4-D dense `CuArray`s (views and other wrappers are not accepted)
in **(head_dim, num_heads, seq_len, batch)** order (head dimension fastest / innermost, as
cuDNN requires). This matches NNlib's internal multi-head attention layout, so no permute is
needed to interoperate with it. With head dim `d`:
* `q`  :  `(d, h, sq, b)`
* `k`  :  `(d, h, skv, b)`
* `v`  :  `(d, h, skv, b)`
* `out`:  `(d, h, sq, b)`  (allocated by the non-`!` form)

# Keyword arguments

* `scale::Real = 1/sqrt(d)`: scalar multiplied into the `QK'` scores before softmax.

# Notes

Forward inference only, in `Float16`/`BFloat16`, with equal head counts and head dimensions
for `q`/`k`/`v` (no GQA/MQA or distinct value head dimension yet), no masking, no dropout,
and no training/backward. The fused engine requires an Ampere (sm_80) or newer GPU.
`Float32`/`Float64` are **not** supported by cuDNN's fused attention engine.

Execution plans are cached per `(eltype, head_dim, sq, skv, num_heads, batch)` on the
calling task's cuDNN handle. The first call for a shape/handle pair pays the one-time plan
build cost; cached plans are freed when the handle is destroyed or evicted under memory
pressure.
"""
cudnnSDPAForward, cudnnSDPAForward!


struct SDPAPlan
    plan::cudnnBackendDescriptor
    workspace_size::Int
end

unsafe_destroy!(plan::SDPAPlan) = unsafe_destroy!(plan.plan)

# fixed tensor UIDs in the graph
const SDPA_UID_Q = 1
const SDPA_UID_K = 2
const SDPA_UID_V = 3
const SDPA_UID_O = 4
const SDPA_UID_SCALE = 5
const SDPA_UIDS = Int64[SDPA_UID_Q, SDPA_UID_K, SDPA_UID_V, SDPA_UID_O, SDPA_UID_SCALE]

# A Julia (head_dim, nheads, seq_len, batch) column-major array (NNlib's internal attention
# layout, a BSHD physical layout) maps to the cuDNN logical tensor [b, h, s, d] with head_dim
# innermost (stride 1). The dedicated SDPA op takes batch and heads as native leading dims.
attn_dims(d, s, h, b) = Int64[b, h, s, d]
attn_strides(d, s, h, b) = Int64[d*h*s, d, d*h, 1]

function build_sdpa_plan(T::DataType, d, sq, skv, h, b)
    dt = cudnnDataType(T)
    F = CUDNN_DATA_FLOAT
    intermediates = cudnnBackendDescriptor[]
    track!(desc) = (push!(intermediates, desc); desc)

    try
        Q = track!(backend_tensor(uid=SDPA_UID_Q, dims=attn_dims(d,sq,h,b),
                                  strides=attn_strides(d,sq,h,b), dtype=dt))
        K = track!(backend_tensor(uid=SDPA_UID_K, dims=attn_dims(d,skv,h,b),
                                  strides=attn_strides(d,skv,h,b), dtype=dt))
        V = track!(backend_tensor(uid=SDPA_UID_V, dims=attn_dims(d,skv,h,b),
                                  strides=attn_strides(d,skv,h,b), dtype=dt))
        O = track!(backend_tensor(uid=SDPA_UID_O, dims=attn_dims(d,sq,h,b),
                                  strides=attn_strides(d,sq,h,b), dtype=dt))
        scale = track!(backend_tensor(uid=SDPA_UID_SCALE, dims=Int64[1,1,1,1],
                                      strides=Int64[1,1,1,1], dtype=F, by_value=true))

        op = track!(cudnnBackendDescriptor(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR))
        setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, Q)
        setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, K)
        setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, V)
        setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, O)
        setattr!(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, scale)
        cudnnBackendFinalize(op)

        graph = track!(operation_graph(cudnnBackendDescriptor[op]))
        deviceprop = track!(backend_deviceprop())
        cfgs = engine_configs(graph; deviceprop)
        append!(intermediates, cfgs)

        for cfg in cfgs
            plan = try_execution_plan(cfg; deviceprop)
            plan === nothing && continue
            try
                return SDPAPlan(plan, Int(plan_workspace_size(plan)))
            catch
                unsafe_destroy!(plan)
                rethrow()
            end
        end
        error("cuDNN: no supported fused-attention engine for T=$T d=$d sq=$sq skv=$skv h=$h b=$b " *
              (isempty(cfgs) ? "(the heuristic returned no engine configurations for this graph)" :
                               "($(length(cfgs)) candidate engine configurations all failed to finalize as unsupported)"))
    finally
        for desc in Iterators.reverse(intermediates)
            unsafe_destroy!(desc)
        end
    end
end

function sdpa_plan(T, d, sq, skv, h, b)
    # Plans are finalized against the current cuDNN handle and are not guaranteed safe for
    # concurrent execution. Cache them with the pooled handle, matching PyTorch's thread-local
    # SDPA graph-cache invariant while still allowing reuse when the handle returns to the pool.
    plans = handle().plans
    key = (:sdpa_fwd, T, d, sq, skv, h, b)
    return get!(plans, key) do
        build_sdpa_plan(T, d, sq, skv, h, b)
    end::SDPAPlan
end


function cudnnSDPAForward(q::DenseCuArray{T,4}, k::DenseCuArray{T,4}, v::DenseCuArray{T,4};
                          o...) where {T}
    out = similar(q)
    cudnnSDPAForward!(out, q, k, v; o...)
end

# The DenseCuArray constraint is essential: the cached plan bakes in dense column-major
# strides, so a non-contiguous view (or a host Array) would pass size checks but make cuDNN
# read the wrong memory.
function cudnnSDPAForward!(out::DenseCuArray{T,4}, q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                           v::DenseCuArray{T,4};
                           scale::Real = 1/sqrt(size(q,1)),
                           is_causal::Bool = false) where {T}
    T in (Float16, BFloat16) ||
        throw(ArgumentError("cudnnSDPAForward only supports Float16/BFloat16, got $T " *
                            "(cuDNN's fused attention engine does not support Float32/Float64)"))
    # Causal masking is not yet supported. On cuDNN <= 9.20 there is no usable path through the
    # raw backend API: the SDPA node's clean score-modifier subgraph
    # (CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH) requires cuDNN >= 9.21 (no CUDNN_jll yet), the
    # block-mask attribute is block-sparse rather than element-wise, and the primitive
    # matmul->softmax->matmul graph yields no fused engine from raw backend calls. Revisit and
    # implement via the score-modifier subgraph once CUDNN_jll >= 9.21 ships.
    is_causal && throw(ArgumentError("causal masking is not yet supported by cudnnSDPAForward"))
    d, h, sq, b = size(q)
    dk, hk, skv, bk = size(k)
    (dk, hk, bk) == (d, h, b) ||
        throw(DimensionMismatch("k must have size (d, h, skv, b) matching q's d, h, b; " *
                                "got $(size(k)), expected ($d, $h, skv, $b)"))
    size(v) == (d, h, skv, b) ||
        throw(DimensionMismatch("v must have size (d, h, skv, b) matching q and k; " *
                                "got $(size(v)), expected $((d, h, skv, b))"))
    size(out) == (d, h, sq, b) ||
        throw(DimensionMismatch("out must have size (d, h, sq, b) matching q; " *
                                "got $(size(out)), expected $((d, h, sq, b))"))
    d % 8 == 0 || throw(ArgumentError("head_dim must be a multiple of 8, got $d"))

    plan = sdpa_plan(T, d, sq, skv, h, b)

    scalebuf = Float32[scale]
    pointers = Any[pointer(q), pointer(k), pointer(v), pointer(out), pointer(scalebuf)]
    with_workspace(plan.workspace_size) do workspace
        vp = variant_pack(uids=SDPA_UIDS, pointers=pointers,
                          workspace=(sizeof(workspace) == 0 ? C_NULL : pointer(workspace)))
        try
            GC.@preserve q k v out scalebuf vp begin
                cudnnBackendExecute(handle(), plan.plan.ptr, vp.ptr)
            end
        finally
            unsafe_destroy!(vp)
        end
    end
    return out
end


@doc raw"""
    cudnnSDPABackward(dout, q, k, v, out; scale, is_causal)

Backward pass for [`cudnnSDPAForward`](@ref), returning the gradients `(dq, dk, dv)` with
respect to `q`, `k`, `v` given the output gradient `dout` and the forward output `out`.

**Not yet implemented.** cuDNN's fused-attention backward is not reachable on
cuDNN <= 9.20 through the backend graph API:

* The dedicated `CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR` (type 43) does not finalize into a
  valid operation graph in this version (`CUDNN_STATUS_BAD_PARAM` with Q/K/V/O/Stats/scale/dO ->
  dQ/dK/dV all set); it is effectively undocumented/incomplete before cuDNN 9.21.
* The documented path is a score-modifier subgraph on the SDPA node
  (`CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH`), with the forward additionally emitting its FP32
  log-sum-exp `stats` (`CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC`, already verified to work).
  It requires cuDNN >= 9.21, for which no `CUDNN_jll` artifact exists yet.

This is the same cuDNN-version limitation that blocks `is_causal=true` in
[`cudnnSDPAForward`](@ref). Once `CUDNN_jll >= 9.21` is available, implement this by having the
forward output the `stats` tensor and building the backward via the SDPA score-modifier
subgraph mechanism.
"""
function cudnnSDPABackward(dout, q, k, v, out; o...)
    error("cudnnSDPABackward is not yet implemented: cuDNN's fused-attention backward requires " *
          "cuDNN >= 9.21 (no CUDNN_jll available yet). See the cudnnSDPABackward docstring.")
end
