# Thin Julia layer over the cuDNN *backend graph API* (the `cudnnBackend*` functions).
#
# Unlike the legacy descriptor API (see descriptors.jl), the backend API is fully generic:
# every object is a `cudnnBackendDescriptor_t` configured through
# `cudnnBackendSetAttribute(desc, name, type, count, ptr)`, finalized with
# `cudnnBackendFinalize`, and read back with `cudnnBackendGetAttribute`. This file provides a
# typed wrapper plus small constructor helpers for the pieces needed to build and run a fused
# graph (tensors, operation graph, engine heuristics, execution plan, variant pack). It is
# currently used by sdpa.jl.


# A finalized-or-not backend descriptor. Owns the handle and destroys it on GC.
mutable struct cudnnBackendDescriptor
    ptr::cudnnBackendDescriptor_t
end

function cudnnBackendDescriptor(descriptorType::cudnnBackendDescriptorType_t)
    ref = Ref{cudnnBackendDescriptor_t}(C_NULL)
    cudnnBackendCreateDescriptor(descriptorType, ref)
    d = cudnnBackendDescriptor(ref[])
    finalizer(x -> cudnnBackendDestroyDescriptor(x.ptr), d)
    return d
end

Base.unsafe_convert(::Type{cudnnBackendDescriptor_t}, d::cudnnBackendDescriptor) = d.ptr

bfinalize!(d::cudnnBackendDescriptor) = (cudnnBackendFinalize(d.ptr); d)


# --- setattr! ---------------------------------------------------------------------------
#
# Set a backend attribute, dispatching on the Julia value type to pick the cuDNN attribute
# type, element count, and host buffer. The buffer is GC-preserved across the ccall.

# core: `buf` is a host array/Ref backing `count` contiguous elements of the attribute type.
function _setattr!(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                   atype::cudnnBackendAttributeType_t, count::Integer, buf)
    GC.@preserve buf begin
        cudnnBackendSetAttribute(d.ptr, name, atype, Int64(count),
                                 convert(Ptr{Cvoid}, pointer(buf)))
    end
    return d
end

# integer dims/strides/ids/sizes
setattr!(d, name, v::Integer) = _setattr!(d, name, CUDNN_TYPE_INT64, 1, Int64[v])
setattr!(d, name, v::AbstractVector{<:Integer}) =
    _setattr!(d, name, CUDNN_TYPE_INT64, length(v), Int64.(collect(v)))

# booleans (cuDNN CUDNN_TYPE_BOOLEAN is a 1-byte bool, matching Julia Bool)
setattr!(d, name, v::Bool) = _setattr!(d, name, CUDNN_TYPE_BOOLEAN, 1, Bool[v])

# doubles (e.g. dropout probability)
setattr!(d, name, v::Float64) = _setattr!(d, name, CUDNN_TYPE_DOUBLE, 1, Float64[v])

# enums: data type, heuristic mode
setattr!(d, name, v::cudnnDataType_t) = _setattr!(d, name, CUDNN_TYPE_DATA_TYPE, 1, [v])
setattr!(d, name, v::cudnnBackendHeurMode_t) = _setattr!(d, name, CUDNN_TYPE_HEUR_MODE, 1, [v])

# the cuDNN handle (accepts the raw handle or the `Handle` wrapper from handle())
setattr_handle!(d, name) =
    _setattr!(d, name, CUDNN_TYPE_HANDLE, 1, cudnnHandle_t[Base.unsafe_convert(cudnnHandle_t, handle())])

# nested descriptor(s)
setattr!(d, name, v::cudnnBackendDescriptor) =
    _setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, cudnnBackendDescriptor_t[v.ptr])
setattr!(d, name, v::AbstractVector{cudnnBackendDescriptor}) =
    _setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, length(v),
              cudnnBackendDescriptor_t[x.ptr for x in v])


# --- getattr ----------------------------------------------------------------------------

# Read up to `maxcount` plain (non-descriptor) elements of type `T`.
function getattr(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                 atype::cudnnBackendAttributeType_t, ::Type{T}, maxcount::Integer) where {T}
    out = Vector{T}(undef, maxcount)
    n = Ref{Int64}(0)
    GC.@preserve out begin
        cudnnBackendGetAttribute(d.ptr, name, atype, Int64(maxcount), n,
                                 convert(Ptr{Cvoid}, pointer(out)))
    end
    resize!(out, n[])
    return out
end

getattr_int64(d, name) = getattr(d, name, CUDNN_TYPE_INT64, Int64, 1)[]

# Read an array of nested descriptors. cuDNN requires the caller to pre-create the output
# descriptors; GetAttribute then populates their contents.
#
# We deliberately avoid creating Julia wrapper objects (with GC finalizers) for the unused
# maxcount-n slots: if a finalizer fires cudnnBackendDestroyDescriptor while another thread
# is inside a cudnnBackendExecute that holds cuDNN's JIT lock, we get a deadlock on bf16
# and other runtime-compiled engines. Instead we use raw handles, destroy the unused ones
# synchronously right here, and only register finalizers for the n handles we return.
function getattr_descriptors(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                             desctype::cudnnBackendDescriptorType_t, maxcount::Integer)
    raw = Vector{cudnnBackendDescriptor_t}(undef, maxcount)
    for i in 1:maxcount
        r = Ref{cudnnBackendDescriptor_t}()
        cudnnBackendCreateDescriptor(desctype, r)
        raw[i] = r[]
    end
    n = Ref{Int64}(0)
    GC.@preserve raw begin
        cudnnBackendGetAttribute(d.ptr, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, Int64(maxcount),
                                 n, convert(Ptr{Cvoid}, pointer(raw)))
    end
    for i in n[]+1:maxcount
        cudnnBackendDestroyDescriptor(raw[i])
    end
    return map(1:n[]) do i
        desc = cudnnBackendDescriptor(raw[i])
        finalizer(x -> cudnnBackendDestroyDescriptor(x.ptr), desc)
        desc
    end
end


# --- operation-node constructor helpers -------------------------------------------------

"""
    backend_tensor(; uid, dims, strides, dtype, is_virtual=false, by_value=false, alignment=16)

Create and finalize a `CUDNN_BACKEND_TENSOR_DESCRIPTOR`. `dims`/`strides` are in cuDNN order
(outermost first, innermost last; the innermost stride is typically 1).
"""
function backend_tensor(; uid::Integer, dims, strides, dtype::cudnnDataType_t,
                        is_virtual::Bool=false, by_value::Bool=false, alignment::Integer=16)
    d = cudnnBackendDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR)
    setattr!(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, Int64(uid))
    setattr!(d, CUDNN_ATTR_TENSOR_DATA_TYPE, dtype)
    setattr!(d, CUDNN_ATTR_TENSOR_DIMENSIONS, dims)
    setattr!(d, CUDNN_ATTR_TENSOR_STRIDES, strides)
    setattr!(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, Int64(alignment))
    is_virtual && setattr!(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, true)
    by_value && setattr!(d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, true)
    return bfinalize!(d)
end

function operation_graph(ops::AbstractVector{cudnnBackendDescriptor})
    g = cudnnBackendDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)
    setattr_handle!(g, CUDNN_ATTR_OPERATIONGRAPH_HANDLE)
    setattr!(g, CUDNN_ATTR_OPERATIONGRAPH_OPS, ops)
    bfinalize!(g)
    return g
end

# Return the engine-config descriptors the heuristic suggests, in preference order.
function engine_configs(graph::cudnnBackendDescriptor;
                        mode::cudnnBackendHeurMode_t=CUDNN_HEUR_MODE_A, maxcount::Integer=16)
    heur = cudnnBackendDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)
    setattr!(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph)
    setattr!(heur, CUDNN_ATTR_ENGINEHEUR_MODE, mode)
    bfinalize!(heur)
    cfgs = getattr_descriptors(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                               CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, maxcount)
    return (heur, cfgs)   # keep `heur` alive: the cfgs reference it
end

# Build and finalize an execution plan for an engine config. Returns `nothing` if cuDNN
# reports the config as not supported (a normal outcome — callers iterate configs until one
# finalizes). Any other error (e.g. BAD_PARAM, which indicates a graph-construction bug
# rather than an unsupported config) is rethrown.
function try_execution_plan(enginecfg::cudnnBackendDescriptor)
    plan = cudnnBackendDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)
    setattr_handle!(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE)
    setattr!(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, enginecfg)
    try
        bfinalize!(plan)
    catch e
        e isa CUDNNError || rethrow()
        # the CUDNN_STATUS_NOT_SUPPORTED family occupies the 3000s
        3000 <= Int(e.code) < 4000 || rethrow()
        # Destroy the descriptor immediately rather than leaving a GC finalizer pending.
        # A finalizer calling cudnnBackendDestroyDescriptor can deadlock against a concurrent
        # cudnnBackendExecute that holds cuDNN's JIT compilation lock (e.g. bf16 SDPA).
        ptr = plan.ptr; plan.ptr = C_NULL
        cudnnBackendDestroyDescriptor(ptr)
        return nothing
    end
    return plan
end

plan_workspace_size(plan) = getattr_int64(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)

# Build and finalize a variant pack. `pointers` are device (or, for by-value tensors, host)
# pointers matching `uids` one-to-one; `workspace` is a device pointer or C_NULL.
function variant_pack(; uids::AbstractVector{<:Integer}, pointers::AbstractVector,
                      workspace)
    vp = cudnnBackendDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)
    ptrbuf = [reinterpret(Ptr{Cvoid}, p) for p in pointers]
    setattr!(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, Int64.(collect(uids)))
    _setattr!(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,
              length(ptrbuf), ptrbuf)
    _setattr!(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
              Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, workspace)])
    bfinalize!(vp)
    return vp
end
