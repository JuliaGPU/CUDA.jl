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

function purge_cache!(ws::Workspace, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    cutensornetWorkspacePurgeCache(handle(), ws, memspace)
    return ws
end


## autotuning

"""
    NetworkAutotunePreferences(; max_iterations=3, intermediate_modes=2)

Settings for [`autotune!`](@ref). `max_iterations` is the number of iterations to time the
candidate kernels for, and `intermediate_modes` whether to also tune the mode order of
intermediate tensors (`0` off, `1` on, `2` lets cuTensorNet decide).
"""
Base.@kwdef struct NetworkAutotunePreferences
    max_iterations::Int32=3
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

@public CuNetwork, append_tensor!, set_output!, attribute, attribute!, tensor_ids,
        rehearse_contraction, set_optimizer_info!, prepare_contraction!, set_gradient!,
        set_adjoint!, autotune!, contract!, purge_cache!, compute_gradients!,
        NetworkAutotunePreferences, pack, unpack!

"""
    CuNetwork{T}()

An empty tensor network with elements of type `T`, to be contracted on the GPU. Add input
tensors with [`append_tensor!`](@ref) and the output tensor with [`set_output!`](@ref). Then
find a contraction path with [`rehearse_contraction`](@ref) (or reuse one with
[`set_optimizer_info!`](@ref)), allocate the workspace with [`prepare_contraction!`](@ref),
and contract the network with [`contract!`](@ref) as often as needed. The network keeps its
tensors alive and uses their memory directly: their contents can change between
contractions, but call [`purge_cache!`](@ref) when that of a `constant` input does.

```julia
net = CuNetwork{Float64}()
append_tensor!(net, A, [1, 2])  # C[i,k] = A[i,j] B[j,k]
append_tensor!(net, B, [2, 3])
set_output!(net, C, [1, 3])
rehearse_contraction(net, 2^30)
prepare_contraction!(net)
contract!(net)
```

Wraps `cutensornetNetworkDescriptor_t`, as created by `cutensornetCreateNetwork`.
"""
mutable struct CuNetwork{T}
    handle::cutensornetNetworkDescriptor_t
    # the descriptor refers to this memory once it has been set. inputs are keyed by the id
    # the library assigned them, which isn't necessarily sequential.
    inputs::Dict{Int64,CuArray{T}}
    output::Union{Nothing,CuArray{T}}
    gradients::Dict{Int64,CuArray{T}}
    adjoint::Union{Nothing,CuArray{T}}
    info::Union{Nothing,CuTensorNetworkContractionOptimizerInfo}
    workspace::Union{Nothing,Workspace}
    function CuNetwork{T}() where {T}
        desc_ref = Ref{cutensornetNetworkDescriptor_t}()
        cutensornetCreateNetwork(handle(), desc_ref)
        obj = new{T}(desc_ref[], Dict{Int64,CuArray{T}}(), nothing, Dict{Int64,CuArray{T}}(), nothing, nothing, nothing)
        resource_finalizer(cutensornetDestroyNetwork, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetNetworkDescriptor_t}, net::CuNetwork) = net.handle
Base.eltype(::CuNetwork{T}) where {T} = T

"""
    append_tensor!(net::CuNetwork{T}, tensor::CuArray{T}, modes;
                   conjugate=false, constant=false, requires_gradient=false) -> Int64

Add `tensor` as an input of the network, with an integer label for each of its modes, and
return the id the network gives it. Modes with the same label are contracted. `conjugate`
uses the complex conjugate of the tensor, `constant` promises that its contents don't change
between contractions so that intermediate results can be cached, and `requires_gradient`
makes [`compute_gradients!`](@ref) compute the gradient with respect to it.

Wraps `cutensornetNetworkAppendTensor`.
"""
function append_tensor!(net::CuNetwork{T}, tensor::CuArray{T}, modes; conjugate::Bool=false, constant::Bool=false, requires_gradient::Bool=false) where {T}
    ndims(tensor) == length(modes) || throw(DimensionMismatch("tensor has $(ndims(tensor)) dimensions, but $(length(modes)) modes were specified."))
    qualifiers = Ref(cutensornetTensorQualifiers_t(conjugate, constant, requires_gradient))
    id = Ref{Int64}()
    cutensornetNetworkAppendTensor(handle(), net, ndims(tensor), collect(Int64, size(tensor)), convert(Vector{Int32}, collect(modes)), qualifiers, T, id)
    net.inputs[id[]] = tensor
    return id[]
end

"""
    set_output!(net::CuNetwork{T}, tensor::CuArray{T}, modes) -> net

Set `tensor` as the output of the network, with an integer label for each of its modes.
Modes of the inputs that are not in `modes` are summed over.

Wraps `cutensornetNetworkSetOutputTensor`.
"""
function set_output!(net::CuNetwork{T}, tensor::CuArray{T}, modes) where {T}
    ndims(tensor) == length(modes) || throw(DimensionMismatch("tensor has $(ndims(tensor)) dimensions, but $(length(modes)) modes were specified."))
    cutensornetNetworkSetOutputTensor(handle(), net, ndims(tensor), convert(Vector{Int32}, collect(modes)), T)
    net.output = tensor
    return net
end

"""
    attribute(net::CuNetwork, attr::cutensornetNetworkAttributes_t, R) -> R

Read an attribute of a network, given by NVIDIA's attribute enum (e.g.
`CUTENSORNET_NETWORK_COMPUTE_TYPE`), as a value of type `R`, the type that NVIDIA documents
for the attribute. This only works for attributes with a scalar value: use
[`tensor_ids`](@ref) for the lists of input tensors.

Wraps `cutensornetNetworkGetAttribute`.
"""
function attribute(net::CuNetwork, attr::cutensornetNetworkAttributes_t, ::Type{R}) where {R}
    attr_buf = Ref{R}()
    cutensornetNetworkGetAttribute(handle(), net, attr, attr_buf, sizeof(R))
    return attr_buf[]
end

"""
    attribute!(net::CuNetwork, attr::cutensornetNetworkAttributes_t, value) -> net
    attribute!(info::CuTensorNetworkContractionOptimizerInfo,
               attr::cutensornetContractionOptimizerInfoAttributes_t, value) -> info

Set an attribute of a network or of a contraction path, given by NVIDIA's attribute enum.
`value` must have the type that NVIDIA documents for the attribute, e.g.
`attribute!(net, CUTENSORNET_NETWORK_COMPUTE_TYPE, CUTENSORNET_COMPUTE_64F)`. A changed
contraction path must be attached again with [`set_optimizer_info!`](@ref).

Wraps `cutensornetNetworkSetAttribute` and `cutensornetContractionOptimizerInfoSetAttribute`.
"""
function attribute!(net::CuNetwork, attr::cutensornetNetworkAttributes_t, value)
    attr_buf = Ref(value)
    cutensornetNetworkSetAttribute(handle(), net, attr, attr_buf, sizeof(attr_buf))
    return net
end

"""
    tensor_ids(net::CuNetwork, num_attr, list_attr) -> Vector{Int32}

The ids of the input tensors with a qualifier, given by the network attributes for their
number and their list, e.g. `CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT` and
`CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT`.
"""
function tensor_ids(net::CuNetwork, num_attr::cutensornetNetworkAttributes_t, list_attr::cutensornetNetworkAttributes_t)
    ids = Vector{Int32}(undef, attribute(net, num_attr, Int32))
    GC.@preserve ids begin
        list = Ref(cutensornetTensorIDList_t(length(ids), pointer(ids)))
        cutensornetNetworkGetAttribute(handle(), net, list_attr, list, sizeof(list))
    end
    return ids
end

"""
    rehearse_contraction(net::CuNetwork, max_workspace_size, config=OptimizerConfig())
        -> CuTensorNetworkContractionOptimizerInfo

Find a contraction path for the network that needs at most `max_workspace_size` bytes of
workspace, slicing the contraction if needed, and attach it to the network. The search is
configured with [`OptimizerConfig`](@ref).

Wraps `cutensornetContractionOptimize`.
"""
function rehearse_contraction(net::CuNetwork, max_workspace_size::Integer, optimizer_conf::OptimizerConfig=OptimizerConfig())
    info = CuTensorNetworkContractionOptimizerInfo(net)
    config = CuTensorNetworkContractionOptimizerConfig(optimizer_conf)
    cutensornetContractionOptimize(handle(), net, config, max_workspace_size, info)
    net.info = info
    return info
end

"""
    set_optimizer_info!(net::CuNetwork, info::CuTensorNetworkContractionOptimizerInfo) -> net

Use the contraction path `info` for the network, e.g. one found for another network of the
same shape, or one changed with [`attribute!`](@ref) or [`unpack!`](@ref).

Wraps `cutensornetNetworkSetOptimizerInfo`.
"""
function set_optimizer_info!(net::CuNetwork, info::CuTensorNetworkContractionOptimizerInfo)
    cutensornetNetworkSetOptimizerInfo(handle(), net, info)
    net.info = info
    return net
end

"""
    prepare_contraction!(net::CuNetwork; gradients=false,
                         preference=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED) -> net

Allocate the workspace for contracting the network along its contraction path, and prepare
the contraction. This is needed before [`autotune!`](@ref) and [`contract!`](@ref), and with
`gradients = true` before [`compute_gradients!`](@ref). `preference` (a
`cutensornetWorksizePref_t`) selects how much workspace to allocate.

Wraps `cutensornetNetworkPrepareContraction` and `cutensornetNetworkPrepareGradientsBackward`.
"""
function prepare_contraction!(net::CuNetwork; gradients::Bool=false, preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED)
    net.info === nothing && throw(ArgumentError("the network has no contraction path; call `rehearse_contraction` or `set_optimizer_info!` first."))
    net.output === nothing && throw(ArgumentError("the network has no output tensor; call `set_output!` first."))
    workspace = Workspace()
    cutensornetWorkspaceComputeContractionSizes(handle(), net, net.info, workspace)
    allocate!(workspace; preference)
    net.workspace = workspace
    cutensornetNetworkPrepareContraction(handle(), net, workspace)
    gradients && cutensornetNetworkPrepareGradientsBackward(handle(), net, workspace)
    for (id, tensor) in net.inputs
        cutensornetNetworkSetInputTensorMemory(handle(), net, id, tensor, C_NULL)
    end
    cutensornetNetworkSetOutputTensorMemory(handle(), net, net.output, C_NULL)
    return net
end

"""
    set_gradient!(net::CuNetwork{T}, id, gradient::CuArray{T}) -> net

Set the array that [`compute_gradients!`](@ref) writes the gradient with respect to input
`id` to. It has the size of that input, which must have been added with
`requires_gradient = true`.

Wraps `cutensornetNetworkSetGradientTensorMemory`.
"""
function set_gradient!(net::CuNetwork{T}, id::Integer, gradient::CuArray{T}) where {T}
    haskey(net.inputs, id) || throw(ArgumentError("the network has no input tensor with id $id."))
    size(gradient) == size(net.inputs[id]) || throw(DimensionMismatch("gradient must have the size of the input tensor."))
    cutensornetNetworkSetGradientTensorMemory(handle(), net, id, gradient, C_NULL)
    net.gradients[id] = gradient
    return net
end

"""
    set_adjoint!(net::CuNetwork{T}, adjoint::CuArray{T}) -> net

Set the adjoint of the output, which seeds the backward pass of [`compute_gradients!`](@ref).
It is the derivative of the final result with respect to the output tensor, and has the
size of the output.

Wraps `cutensornetNetworkSetAdjointTensorMemory`.
"""
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

"""
    autotune!(net::CuNetwork; prefs=NetworkAutotunePreferences(), stream=stream()) -> net

Time the kernels for each pairwise contraction of the prepared network on its tensors, and
use the fastest ones in later contractions. See [`NetworkAutotunePreferences`](@ref).

Wraps `cutensornetNetworkAutotuneContraction`.
"""
function autotune!(net::CuNetwork; prefs::Union{NetworkAutotunePreferences,CuNetworkAutotunePreference}=NetworkAutotunePreferences(), stream::CuStream=stream())
    CUDACore.check_owned(stream)
    if prefs isa NetworkAutotunePreferences
        prefs = CuNetworkAutotunePreference(prefs)
    end
    cutensornetNetworkAutotuneContraction(handle(), net, require_prepared(net), prefs, stream)
    return net
end

"""
    contract!(net::CuNetwork; accumulate=false, slices=nothing, stream=stream()) -> output

Contract the prepared network into its output tensor, which is returned. With `accumulate`,
the result is added to the output. `slices`, a [`CuTensorNetworkSliceGroup`](@ref), limits
the contraction to some of the slices, e.g. to distribute it.

Wraps `cutensornetNetworkContract`.
"""
function contract!(net::CuNetwork; accumulate::Bool=false, slices::Union{Nothing,CuTensorNetworkSliceGroup}=nothing, stream::CuStream=stream())
    CUDACore.check_owned(stream)
    slice_group = slices === nothing ? C_NULL : slices.handle
    cutensornetNetworkContract(handle(), net, accumulate, require_prepared(net), slice_group, stream)
    return net.output
end

"""
    purge_cache!(net::CuNetwork) -> net

Drop the intermediate results cached because of inputs added with `constant = true`. Call
this after changing the contents of such an input.

Wraps `cutensornetWorkspacePurgeCache`.
"""
function purge_cache!(net::CuNetwork, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    purge_cache!(require_prepared(net), memspace)
    return net
end

"""
    compute_gradients!(net::CuNetwork; accumulate=false, slices=nothing, stream=stream())
        -> Dict{Int64,CuArray}

Compute the gradients of the contraction with respect to the inputs added with
`requires_gradient = true`, after a call to [`contract!`](@ref). They are written to the
arrays set with [`set_gradient!`](@ref), for the adjoint set with [`set_adjoint!`](@ref),
and returned by input id. The network must have been prepared with `gradients = true`.

Wraps `cutensornetNetworkComputeGradientsBackward`.
"""
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

"""
    pack(info::CuTensorNetworkContractionOptimizerInfo) -> Vector{UInt8}

Serialize a contraction path, so that it can be reused without searching again, with
[`unpack!`](@ref) or [`CuTensorNetworkContractionOptimizerInfo`](@ref).

Wraps `cutensornetContractionOptimizerInfoPackData`.
"""
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

"""
    unpack!(info::CuTensorNetworkContractionOptimizerInfo, data::Vector{UInt8}) -> info

Replace the contraction path in `info` by one serialized with [`pack`](@ref). A path that
belongs to a network must be attached to it again with [`set_optimizer_info!`](@ref).

Wraps `cutensornetUpdateContractionOptimizerInfoFromPackedData`.
"""
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
