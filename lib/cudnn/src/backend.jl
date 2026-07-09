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
    finalizer(unsafe_destroy!, d)
    return d
end

Base.unsafe_convert(::Type{cudnnBackendDescriptor_t}, d::cudnnBackendDescriptor) = d.ptr

function unsafe_destroy!(d::cudnnBackendDescriptor)
    ptr = d.ptr
    ptr == C_NULL && return
    d.ptr = C_NULL
    cudnnBackendDestroyDescriptor(ptr)
    return
end


# --- setattr! ---------------------------------------------------------------------------
#
# Set a backend attribute, dispatching on the Julia value type to pick the cuDNN attribute
# type, element count, and host buffer. The buffer is GC-preserved across the ccall.

# core: `buf` is a host array/Ref backing `count` contiguous elements of the attribute type.
function setattr!(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                  atype::cudnnBackendAttributeType_t, count::Integer, buf)
    GC.@preserve buf begin
        cudnnBackendSetAttribute(d.ptr, name, atype, Int64(count),
                                 convert(Ptr{Cvoid}, pointer(buf)))
    end
    return d
end

# integer dims/strides/ids/sizes
setattr!(d, name, v::Integer) = setattr!(d, name, CUDNN_TYPE_INT64, 1, Int64[v])
setattr!(d, name, v::AbstractVector{<:Integer}) =
    setattr!(d, name, CUDNN_TYPE_INT64, length(v), convert(Vector{Int64}, v))

# booleans (cuDNN CUDNN_TYPE_BOOLEAN is a 1-byte bool, matching Julia Bool)
setattr!(d, name, v::Bool) = setattr!(d, name, CUDNN_TYPE_BOOLEAN, 1, Bool[v])

# doubles (e.g. dropout probability)
setattr!(d, name, v::Float64) = setattr!(d, name, CUDNN_TYPE_DOUBLE, 1, Float64[v])

# enums: data type, heuristic mode
setattr!(d, name, v::cudnnDataType_t) = setattr!(d, name, CUDNN_TYPE_DATA_TYPE, 1, [v])
setattr!(d, name, v::cudnnBackendHeurMode_t) =
    setattr!(d, name, CUDNN_TYPE_HEUR_MODE, 1, [v])

# the cuDNN handle (accepts the raw handle or the `Handle` wrapper from handle())
setattr_handle!(d, name) =
    setattr!(d, name, CUDNN_TYPE_HANDLE, 1,
             cudnnHandle_t[Base.unsafe_convert(cudnnHandle_t, handle())])

# nested descriptor(s)
setattr!(d, name, v::cudnnBackendDescriptor) =
    setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, cudnnBackendDescriptor_t[v.ptr])
setattr!(d, name, v::AbstractVector{cudnnBackendDescriptor}) =
    setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, length(v),
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

function getattr_count(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                       atype::cudnnBackendAttributeType_t)
    n = Ref{Int64}(0)
    cudnnBackendGetAttribute(d.ptr, name, atype, Int64(0), n, C_NULL)
    return n[]
end

# Read an array of nested descriptors. cuDNN requires the caller to pre-create the output
# descriptors; GetAttribute then populates their contents.
#
# Keep raw handles until we know how many cuDNN returned, then destroy the unused descriptors
# synchronously. Julia finalizers are a last-resort cleanup path for descriptors that escape.
function getattr_descriptors(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                             desctype::cudnnBackendDescriptorType_t, maxcount::Integer)
    maxcount == 0 && return cudnnBackendDescriptor[]
    raw = fill(cudnnBackendDescriptor_t(C_NULL), maxcount)
    n = Ref{Int64}(0)
    try
        for i in 1:maxcount
            r = Ref{cudnnBackendDescriptor_t}(C_NULL)
            cudnnBackendCreateDescriptor(desctype, r)
            raw[i] = r[]
        end
        GC.@preserve raw begin
            cudnnBackendGetAttribute(d.ptr, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, Int64(maxcount),
                                     n, convert(Ptr{Cvoid}, pointer(raw)))
        end
    catch
        for ptr in raw
            ptr == C_NULL || cudnnBackendDestroyDescriptor(ptr)
        end
        rethrow()
    end
    nreturned = min(n[], Int64(maxcount))
    for i in nreturned+1:maxcount
        cudnnBackendDestroyDescriptor(raw[i])
    end
    nreturned == 0 && return cudnnBackendDescriptor[]
    return map(1:nreturned) do i
        desc = cudnnBackendDescriptor(raw[i])
        finalizer(unsafe_destroy!, desc)
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
    try
        setattr!(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, Int64(uid))
        setattr!(d, CUDNN_ATTR_TENSOR_DATA_TYPE, dtype)
        setattr!(d, CUDNN_ATTR_TENSOR_DIMENSIONS, dims)
        setattr!(d, CUDNN_ATTR_TENSOR_STRIDES, strides)
        setattr!(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, Int64(alignment))
        is_virtual && setattr!(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, true)
        by_value && setattr!(d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, true)
        cudnnBackendFinalize(d)
        return d
    catch
        unsafe_destroy!(d)
        rethrow()
    end
end

function backend_deviceprop()
    d = cudnnBackendDescriptor(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR)
    try
        setattr_handle!(d, CUDNN_ATTR_DEVICEPROP_HANDLE)
        cudnnBackendFinalize(d)
        return d
    catch
        unsafe_destroy!(d)
        rethrow()
    end
end

function operation_graph(ops::AbstractVector{cudnnBackendDescriptor})
    g = cudnnBackendDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)
    try
        setattr_handle!(g, CUDNN_ATTR_OPERATIONGRAPH_HANDLE)
        setattr!(g, CUDNN_ATTR_OPERATIONGRAPH_OPS, ops)
        cudnnBackendFinalize(g)
        return g
    catch
        unsafe_destroy!(g)
        rethrow()
    end
end

# Return caller-owned engine-config descriptors the heuristic suggests, in preference order.
function engine_configs(graph::cudnnBackendDescriptor;
                        deviceprop::Union{Nothing,cudnnBackendDescriptor}=nothing,
                        mode::cudnnBackendHeurMode_t=CUDNN_HEUR_MODE_A, maxcount::Integer=16)
    heur = cudnnBackendDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)
    try
        setattr!(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph)
        setattr!(heur, CUDNN_ATTR_ENGINEHEUR_MODE, mode)
        deviceprop !== nothing && setattr!(heur, CUDNN_ATTR_ENGINEHEUR_DEVICEPROP, deviceprop)
        cudnnBackendFinalize(heur)
        count = min(Int64(maxcount), getattr_count(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR))
        return getattr_descriptors(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                   CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, count)
    finally
        unsafe_destroy!(heur)
    end
end

# Build and finalize an execution plan for an engine config. Returns `nothing` if cuDNN
# reports the config as not supported (a normal outcome: callers iterate configs until one
# finalizes). Any other error (e.g. BAD_PARAM, which indicates a graph-construction bug
# rather than an unsupported config) is rethrown.
function try_execution_plan(enginecfg::cudnnBackendDescriptor;
                            deviceprop::Union{Nothing,cudnnBackendDescriptor}=nothing)
    plan = cudnnBackendDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)
    try
        setattr_handle!(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE)
        setattr!(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, enginecfg)
        deviceprop !== nothing && setattr!(plan, CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP, deviceprop)
        cudnnBackendFinalize(plan)
    catch e
        # Destroy failed descriptors immediately rather than leaving a GC finalizer pending.
        unsafe_destroy!(plan)
        e isa CUDNNError || rethrow()
        # the CUDNN_STATUS_NOT_SUPPORTED family occupies the 3000s
        3000 <= Int(e.code) < 4000 || rethrow()
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
    try
        ptrbuf = [reinterpret(Ptr{Cvoid}, p) for p in pointers]
        setattr!(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, uids)
        setattr!(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,
                 length(ptrbuf), ptrbuf)
        setattr!(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
                 Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, workspace)])
        cudnnBackendFinalize(vp)
        return vp
    catch
        unsafe_destroy!(vp)
        rethrow()
    end
end
