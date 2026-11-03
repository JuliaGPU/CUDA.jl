# the network API, replacing the deprecated `CuTensorNetworkDescriptor` and contraction plans

## workspaces

# a workspace descriptor together with the memory attached to it. the library's descriptors
# and contraction state refer to this memory, so it needs to outlive their use.
mutable struct Workspace
    desc::CuTensorNetworkWorkspaceDescriptor
    device_scratch::CuVector{UInt8}
    device_cache::CuVector{UInt8}
    host_scratch::Vector{UInt8}
    Workspace() = new(CuTensorNetworkWorkspaceDescriptor(), CuVector{UInt8}(undef, 0),
                      CuVector{UInt8}(undef, 0), UInt8[])
end
Base.unsafe_convert(::Type{cutensornetWorkspaceDescriptor_t}, ws::Workspace) =
    Base.unsafe_convert(cutensornetWorkspaceDescriptor_t, ws.desc)

function workspace_size(ws::Workspace, memspace::cutensornetMemspace_t, kind::cutensornetWorkspaceKind_t;
                        preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED)
    size = Ref{Int64}(0)
    cutensornetWorkspaceGetMemorySize(handle(), ws, preference, memspace, kind, size)
    return Int(size[])
end

# attach as much memory as the library asks for after sizing the workspace
function allocate!(ws::Workspace; preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED)
    function attach(memspace, kind, buffer)
        isempty(buffer) && return
        cutensornetWorkspaceSetMemory(handle(), ws, memspace, kind, buffer, sizeof(buffer))
    end
    ws.device_scratch = CuVector{UInt8}(undef, workspace_size(ws, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH; preference))
    attach(CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, ws.device_scratch)
    ws.device_cache = CuVector{UInt8}(undef, workspace_size(ws, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE; preference))
    attach(CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, ws.device_cache)
    ws.host_scratch = Vector{UInt8}(undef, workspace_size(ws, CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH; preference))
    attach(CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, ws.host_scratch)
    return ws
end

# must be called when the input data of a contraction with a cache has changed
function purge_cache!(ws::Workspace, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    cutensornetWorkspacePurgeCache(handle(), ws, memspace)
    return ws
end


## autotuning

Base.@kwdef struct NetworkAutotunePreferences
    max_iterations::Int32=3
    # 0 is off, 1 is on, and 2 lets the library decide
    intermediate_modes::Int32=2
end

mutable struct CuNetworkAutotunePreference
    handle::cutensornetNetworkAutotunePreference_t
    function CuNetworkAutotunePreference(prefs::NetworkAutotunePreferences=NetworkAutotunePreferences())
        pref_ref = Ref{cutensornetNetworkAutotunePreference_t}()
        cutensornetCreateNetworkAutotunePreference(handle(), pref_ref)
        obj = new(pref_ref[])
        resource_finalizer(cutensornetDestroyNetworkAutotunePreference, obj)
        for (attr, value) in (CUTENSORNET_NETWORK_AUTOTUNE_MAX_ITERATIONS => prefs.max_iterations,
                              CUTENSORNET_NETWORK_AUTOTUNE_INTERMEDIATE_MODES => prefs.intermediate_modes)
            attr_buf = Ref(value)
            cutensornetNetworkAutotunePreferenceSetAttribute(handle(), obj, attr, attr_buf, sizeof(attr_buf))
        end
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetNetworkAutotunePreference_t}, prefs::CuNetworkAutotunePreference) = prefs.handle

function attribute(prefs::CuNetworkAutotunePreference, attr::cutensornetNetworkAutotunePreferenceAttributes_t)
    attr_buf = Ref{Int32}(0)
    cutensornetNetworkAutotunePreferenceGetAttribute(handle(), prefs, attr, attr_buf, sizeof(attr_buf))
    return attr_buf[]
end


## networks

mutable struct CuNetwork{T}
    handle::cutensornetNetworkDescriptor_t
    # the descriptor refers to this memory once it has been set
    inputs::Vector{CuArray{T}}
    output::Union{Nothing,CuArray{T}}
    gradients::Dict{Int64,CuArray{T}}
    adjoint::Union{Nothing,CuArray{T}}
    info::Union{Nothing,CuTensorNetworkContractionOptimizerInfo}
    workspace::Union{Nothing,Workspace}
    function CuNetwork{T}() where {T}
        desc_ref = Ref{cutensornetNetworkDescriptor_t}()
        cutensornetCreateNetwork(handle(), desc_ref)
        obj = new{T}(desc_ref[], CuArray{T}[], nothing, Dict{Int64,CuArray{T}}(), nothing, nothing, nothing)
        resource_finalizer(cutensornetDestroyNetwork, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetNetworkDescriptor_t}, net::CuNetwork) = net.handle
Base.eltype(::CuNetwork{T}) where {T} = T

function append_tensor!(net::CuNetwork{T}, tensor::CuArray{T}, modes; conjugate::Bool=false, constant::Bool=false, requires_gradient::Bool=false) where {T}
    ndims(tensor) == length(modes) || throw(DimensionMismatch("tensor has $(ndims(tensor)) dimensions, but $(length(modes)) modes were specified."))
    qualifiers = Ref(cutensornetTensorQualifiers_t(conjugate, constant, requires_gradient))
    id = Ref{Int64}()
    cutensornetNetworkAppendTensor(handle(), net, ndims(tensor), collect(Int64, size(tensor)), convert(Vector{Int32}, collect(modes)), qualifiers, T, id)
    push!(net.inputs, tensor)
    return id[]
end

function set_output!(net::CuNetwork{T}, tensor::CuArray{T}, modes) where {T}
    ndims(tensor) == length(modes) || throw(DimensionMismatch("tensor has $(ndims(tensor)) dimensions, but $(length(modes)) modes were specified."))
    cutensornetNetworkSetOutputTensor(handle(), net, ndims(tensor), convert(Vector{Int32}, collect(modes)), T)
    net.output = tensor
    return net
end

function attribute(net::CuNetwork, attr::cutensornetNetworkAttributes_t, ::Type{R}) where {R}
    attr_buf = Ref{R}()
    cutensornetNetworkGetAttribute(handle(), net, attr, attr_buf, sizeof(R))
    return attr_buf[]
end

function attribute!(net::CuNetwork, attr::cutensornetNetworkAttributes_t, value)
    attr_buf = Ref(value)
    cutensornetNetworkSetAttribute(handle(), net, attr, attr_buf, sizeof(attr_buf))
    return net
end

# the ids of the input tensors with a certain qualifier, e.g. `CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT`
function tensor_ids(net::CuNetwork, num_attr::cutensornetNetworkAttributes_t, list_attr::cutensornetNetworkAttributes_t)
    ids = Vector{Int32}(undef, attribute(net, num_attr, Int32))
    GC.@preserve ids begin
        list = Ref(cutensornetTensorIDList_t(length(ids), pointer(ids)))
        cutensornetNetworkGetAttribute(handle(), net, list_attr, list, sizeof(list))
    end
    return ids
end

function rehearse_contraction(net::CuNetwork, max_workspace_size::Integer, optimizer_conf::OptimizerConfig=OptimizerConfig())
    info = CuTensorNetworkContractionOptimizerInfo(net)
    config = CuTensorNetworkContractionOptimizerConfig(optimizer_conf)
    cutensornetContractionOptimize(handle(), net, config, max_workspace_size, info)
    net.info = info
    return info
end

# reuse a contraction path, e.g. one computed for another network of the same shape
function set_optimizer_info!(net::CuNetwork, info::CuTensorNetworkContractionOptimizerInfo)
    cutensornetNetworkSetOptimizerInfo(handle(), net, info)
    net.info = info
    return net
end

function prepare_contraction!(net::CuNetwork; gradients::Bool=false, preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED)
    net.info === nothing && throw(ArgumentError("the network has no contraction path; call `rehearse_contraction` or `set_optimizer_info!` first."))
    net.output === nothing && throw(ArgumentError("the network has no output tensor; call `set_output!` first."))
    workspace = Workspace()
    cutensornetWorkspaceComputeContractionSizes(handle(), net, net.info, workspace)
    allocate!(workspace; preference)
    net.workspace = workspace
    cutensornetNetworkPrepareContraction(handle(), net, workspace)
    gradients && cutensornetNetworkPrepareGradientsBackward(handle(), net, workspace)
    for (id, tensor) in enumerate(net.inputs)
        cutensornetNetworkSetInputTensorMemory(handle(), net, id - 1, tensor, C_NULL)
    end
    cutensornetNetworkSetOutputTensorMemory(handle(), net, net.output, C_NULL)
    return net
end

# the buffer that receives the gradient with respect to the input with the given id
function set_gradient!(net::CuNetwork{T}, id::Integer, gradient::CuArray{T}) where {T}
    size(gradient) == size(net.inputs[id + 1]) || throw(DimensionMismatch("gradient must have the size of the input tensor."))
    cutensornetNetworkSetGradientTensorMemory(handle(), net, id, gradient, C_NULL)
    net.gradients[id] = gradient
    return net
end

# the seed of the backward pass, with the size of the output tensor
function set_adjoint!(net::CuNetwork{T}, adjoint::CuArray{T}) where {T}
    size(adjoint) == size(net.output) || throw(DimensionMismatch("adjoint must have the size of the output tensor."))
    cutensornetNetworkSetAdjointTensorMemory(handle(), net, adjoint, C_NULL)
    net.adjoint = adjoint
    return net
end

function require_prepared(net::CuNetwork)
    net.workspace === nothing && throw(ArgumentError("the network has not been prepared; call `prepare_contraction!` first."))
    return net.workspace
end

function autotune!(net::CuNetwork; prefs::CuNetworkAutotunePreference=CuNetworkAutotunePreference(), stream::CuStream=stream())
    CUDACore.check_owned(stream)
    cutensornetNetworkAutotuneContraction(handle(), net, require_prepared(net), prefs, stream)
    return net
end

function contract!(net::CuNetwork; accumulate::Bool=false, slices::Union{Nothing,CuTensorNetworkSliceGroup}=nothing, stream::CuStream=stream())
    CUDACore.check_owned(stream)
    slice_group = slices === nothing ? C_NULL : slices.handle
    cutensornetNetworkContract(handle(), net, accumulate, require_prepared(net), slice_group, stream)
    return net.output
end

function compute_gradients!(net::CuNetwork; accumulate::Bool=false, slices::Union{Nothing,CuTensorNetworkSliceGroup}=nothing, stream::CuStream=stream())
    CUDACore.check_owned(stream)
    slice_group = slices === nothing ? C_NULL : slices.handle
    cutensornetNetworkComputeGradientsBackward(handle(), net, accumulate, require_prepared(net), slice_group, stream)
    return net.gradients
end


## optimizer info and config attributes

function attribute(config::CuTensorNetworkContractionOptimizerConfig, attr::cutensornetContractionOptimizerConfigAttributes_t, ::Type{R}) where {R}
    attr_buf = Ref{R}()
    cutensornetContractionOptimizerConfigGetAttribute(handle(), config, attr, attr_buf, sizeof(R))
    return attr_buf[]
end

function attribute!(info::CuTensorNetworkContractionOptimizerInfo, attr::cutensornetContractionOptimizerInfoAttributes_t, value)
    attr_buf = Ref(value)
    cutensornetContractionOptimizerInfoSetAttribute(handle(), info, attr, attr_buf, sizeof(attr_buf))
    return info
end

# serialization of a contraction path, so that it can be reused without optimizing again
function pack(info::CuTensorNetworkContractionOptimizerInfo)
    size = Ref{Csize_t}()
    cutensornetContractionOptimizerInfoGetPackedSize(handle(), info, size)
    data = Vector{UInt8}(undef, size[])
    cutensornetContractionOptimizerInfoPackData(handle(), info, data, sizeof(data))
    return data
end

function CuTensorNetworkContractionOptimizerInfo(net, data::Vector{UInt8})
    info_ref = Ref{cutensornetContractionOptimizerInfo_t}()
    cutensornetCreateContractionOptimizerInfoFromPackedData(handle(), net, data, sizeof(data), info_ref)
    return CuTensorNetworkContractionOptimizerInfo(info_ref[])
end

function unpack!(info::CuTensorNetworkContractionOptimizerInfo, data::Vector{UInt8})
    cutensornetUpdateContractionOptimizerInfoFromPackedData(handle(), data, sizeof(data), info)
    return info
end


## tensor descriptor attributes

function is_distributed(desc::CuTensorDescriptor)
    attr_buf = Ref{Int32}()
    cutensornetTensorDescriptorGetAttribute(handle(), desc, CUTENSORNET_TENSOR_DESCRIPTOR_IS_DISTRIBUTED, attr_buf, sizeof(attr_buf))
    return attr_buf[] != 0
end

function local_data_size(desc::CuTensorDescriptor)
    attr_buf = Ref{Csize_t}()
    cutensornetTensorDescriptorGetAttribute(handle(), desc, CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_DATA_SIZE, attr_buf, sizeof(attr_buf))
    return Int(attr_buf[])
end

function local_extents(desc::CuTensorDescriptor)
    extents = Vector{Int64}(undef, ndims(desc))
    cutensornetTensorDescriptorGetAttribute(handle(), desc, CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_EXTENTS, extents, sizeof(extents))
    return tuple(extents...)
end


## errors

# the message for the last error that happened on the calling thread
last_error() = unsafe_string(cutensornetGetLastError())
error_string(status::cutensornetStatus_t) = unsafe_string(cutensornetGetErrorString(status))
