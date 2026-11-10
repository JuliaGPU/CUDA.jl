# Deprecated in v6.6: the contraction API built on network descriptors and contraction
# plans, which NVIDIA deprecated in favor of the API that `CuNetwork` wraps.

mutable struct CuTensorNetworkDescriptor
    handle::cutensornetNetworkDescriptor_t
    function CuTensorNetworkDescriptor(numInputs::Int32, numModesIn::Vector{Int32}, extentsIn::Vector{Vector{Int64}},
                                       stridesIn, modesIn::Vector{Vector{Int32}}, qualifiersIn::Vector{cutensornetTensorQualifiers_t},
                                       numModesOut::Int32, extentsOut::Vector{Int64}, stridesOut::Union{Ptr{Nothing}, Vector{Int64}},
                                       modesOut::Vector{Int32}, dataType::Type, computeType::Type)
        desc_ref = Ref{cutensornetNetworkDescriptor_t}()
        cutensornetCreateNetworkDescriptor(handle(), numInputs, numModesIn, extentsIn, stridesIn, modesIn, qualifiersIn, numModesOut,
                                           extentsOut, stridesOut, modesOut, dataType, computeType, desc_ref)
        obj = new(desc_ref[])
        resource_finalizer(cutensornetDestroyNetworkDescriptor, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetNetworkDescriptor_t}, desc::CuTensorNetworkDescriptor) = desc.handle

function compute_type(T::DataType)
    if T == Float16
        return Float16
    elseif T == Float32
        return Float32
    elseif T == Float64
        return Float64
    end
end

mutable struct CuTensorNetwork{T}
    desc::CuTensorNetworkDescriptor
    input_modes::Vector{Vector{Int32}}
    input_extents::Vector{Vector{Int32}}
    input_strides::Vector{<:Union{Ptr{Nothing}, Vector{Int64}}}
    input_qualifiers::Vector{cutensornetTensorQualifiers_t}
    input_arrs::Vector{CuArray{T}}
    output_modes::Vector{Int32}
    output_extents::Vector{Int32}
    output_strides::Union{Ptr{Nothing}, Vector{Int64}}
    output_arr::CuArray{T}
end
function CuTensorNetwork(T::DataType, input_modes, input_extents, input_strides, input_qualifiers, output_modes, output_extents, output_strides)
    Base.depwarn("`CuTensorNetwork` is deprecated, use `CuNetwork` with `append_tensor!` and `set_output!` instead.", :CuTensorNetwork)
    input_qualifiers = [cutensornetTensorQualifiers_t(isConjugate, 0, 0) for isConjugate in input_qualifiers]
    # TODO: expose isConstant & requiresGradient?
    desc = CuTensorNetworkDescriptor(Int32(length(input_modes)), Int32.(length.(input_modes)), input_extents, input_strides, input_modes, input_qualifiers,
                                     Int32(length(output_modes)), output_extents, output_strides, output_modes, T, compute_type(real(T)))

    return CuTensorNetwork{T}(desc, input_modes, input_extents, input_strides, input_qualifiers,
                              Vector{CuArray{T}}(undef, 0), output_modes, output_extents, output_strides, CUDACore.zeros(T, 0))
end

mutable struct CuTensorNetworkContractionPlan
    handle::cutensornetContractionPlan_t
    function CuTensorNetworkContractionPlan(net_desc::CuTensorNetworkDescriptor, info::CuTensorNetworkContractionOptimizerInfo, ws_desc::CuTensorNetworkWorkspaceDescriptor)
        desc_ref = Ref{cutensornetContractionPlan_t}()
        cutensornetCreateContractionPlan(handle(), net_desc, info, ws_desc, desc_ref)
        obj = new(desc_ref[])
        resource_finalizer(cutensornetDestroyContractionPlan, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cutensornetContractionPlan_t}, desc::CuTensorNetworkContractionPlan) = desc.handle

abstract type UseAutotuning end
struct NoAutoTune <: UseAutotuning end
struct AutoTune   <: UseAutotuning end

Base.@kwdef struct AutotunePreferences
    max_iterations::Int32=3
end

mutable struct CuTensorNetworkAutotunePreference
    handle::cutensornetContractionAutotunePreference_t
    function CuTensorNetworkAutotunePreference(prefs::AutotunePreferences)
        pref_ref = Ref{cutensornetContractionAutotunePreference_t}()
        cutensornetCreateContractionAutotunePreference(handle(), pref_ref)
        obj = new(pref_ref[])
        resource_finalizer(cutensornetDestroyContractionAutotunePreference, obj)
        # apply preference options
        for attr in (:max_iterations=>CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,)
            attr_buf = Ref(Base.getproperty(prefs, attr[1]))
            cutensornetContractionAutotunePreferenceSetAttribute(handle(), pref_ref[], attr[2], attr_buf, sizeof(attr_buf))
        end
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetContractionAutotunePreference_t}, prefs::CuTensorNetworkAutotunePreference)   = prefs.handle

function rehearse_contraction(tn::CuTensorNetwork, max_workspace_size::Integer, optimizer_conf::OptimizerConfig=OptimizerConfig())
    info = CuTensorNetworkContractionOptimizerInfo(tn.desc)
    # set optimizer options
    config = CuTensorNetworkContractionOptimizerConfig(optimizer_conf)
    # perform optimization
    cutensornetContractionOptimize(handle(), tn.desc, config, max_workspace_size, info)
    return info
end

function perform_contraction!(tn::CuTensorNetwork, info, ::NoAutoTune; prefs::AutotunePreferences=AutotunePreferences(), stream::CuStream=stream(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    Base.depwarn("`perform_contraction!` is deprecated, use `prepare_contraction!`, `autotune!` and `contract!` on a `CuNetwork` instead.", :perform_contraction!)
    # (the arrays and the workspace are used as if on the task's stream)
    CUDACore.check_owned(stream)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    cutensornetWorkspaceComputeSizes(handle(), tn.desc, info, workspace_desc)
    actual_ws_size = Ref{UInt64}()
    cutensornetWorkspaceGetSize(handle(), workspace_desc, workspace_preference, memspace, actual_ws_size)
    inputs = tn.input_arrs
    output = tn.output_arr
    cutensornetWorkspaceSet(handle(), workspace_desc, memspace, C_NULL, actual_ws_size[])
    plan = CuTensorNetworkContractionPlan(tn.desc, info, workspace_desc)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])
        input_ptrs = [pointer(arr) for arr in inputs]
        output_ptr = pointer(output)
        slice_group = CuTensorNetworkSliceGroup(0, num_slices(info), 1)
        cutensornetContractSlices(handle(), plan, input_ptrs, output_ptr, 0, workspace_desc, slice_group, stream)
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, C_NULL, 0)
    end
    return tn
end

function perform_contraction!(tn::CuTensorNetwork, info, ::AutoTune; prefs::AutotunePreferences=AutotunePreferences(), stream::CuStream=stream(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    Base.depwarn("`perform_contraction!` is deprecated, use `prepare_contraction!`, `autotune!` and `contract!` on a `CuNetwork` instead.", :perform_contraction!)
    # (the arrays and the workspace are used as if on the task's stream)
    CUDACore.check_owned(stream)
    ctn_prefs = CuTensorNetworkAutotunePreference(prefs)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    cutensornetWorkspaceComputeSizes(handle(), tn.desc, info, workspace_desc)
    inputs = tn.input_arrs
    output = tn.output_arr
    actual_ws_size = Ref{UInt64}(0)
    cutensornetWorkspaceGetSize(handle(), workspace_desc, workspace_preference, memspace, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])
        plan = CuTensorNetworkContractionPlan(tn.desc, info, workspace_desc)
        cutensornetContractionAutotune(handle(), plan, pointer.(tn.input_arrs),
                                       pointer(tn.output_arr), workspace_desc,
                                       ctn_prefs, stream)
        input_ptrs = [pointer(arr) for arr in inputs]
        output_ptr = pointer(output)
        slice_group = CuTensorNetworkSliceGroup(0, num_slices(info), 1)
        cutensornetContractSlices(handle(), plan, input_ptrs, output_ptr, 0, workspace_desc, slice_group, stream)
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, C_NULL, 0)
    end
    return tn
end


# Deprecated in v6.6: tensor decompositions extended LinearAlgebra's `qr!` and `svd!` with
# signatures that mean something else there, and `gateSplit!` was the only camel-case name.

@deprecate LinearAlgebra.qr!(tensor_in::CuArray{T,N}, modes_in, tensor_q::CuArray{T,Q}, modes_q, tensor_r::CuArray{T,R}, modes_r; kwargs...) where {T<:Number,N,Q,R} tensor_qr!(tensor_in, modes_in, tensor_q, modes_q, tensor_r, modes_r; kwargs...) false
@deprecate LinearAlgebra.qr!(tensor_in::CuTensor{T,N}, tensor_q::CuTensor{T,Q}, tensor_r::CuTensor{T,R}; kwargs...) where {T<:Number,N,Q,R} tensor_qr!(tensor_in, tensor_q, tensor_r; kwargs...) false
@deprecate LinearAlgebra.svd!(tensor_in::CuArray{T,N}, modes_in, tensor_u::CuArray{T,U}, modes_u, s::CuVector{S}, tensor_v::CuArray{T,V}, modes_v; kwargs...) where {T<:Number,S<:Real,N,U,V} tensor_svd!(tensor_in, modes_in, tensor_u, modes_u, s, tensor_v, modes_v; kwargs...) false
@deprecate LinearAlgebra.svd!(tensor_in::CuTensor{T,N}, tensor_u::CuTensor{T,U}, s::CuVector{S}, tensor_v::CuTensor{T,V}; kwargs...) where {T<:Number,S<:Real,N,U,V} tensor_svd!(tensor_in, tensor_u, s, tensor_v; kwargs...) false

# the `gateAlgo` keyword is called `algo` now
function gateSplit!(A::CuArray{T,NA}, modes_a, B::CuArray{T,NB}, modes_b, G::CuArray{T,NG}, modes_g, tensor_u::CuArray{T,U}, modes_u, s::CuVector{S}, tensor_v::CuArray{T,V}, modes_v;
                    gateAlgo::cutensornetGateSplitAlgo_t=CUTENSORNET_GATE_SPLIT_ALGO_DIRECT, kwargs...) where {T<:Number,S<:Real,NA,NB,NG,U,V}
    Base.depwarn("`gateSplit!(...; gateAlgo)` is deprecated, use `gate_split!(...; algo)` instead.", :gateSplit!)
    gate_split!(A, modes_a, B, modes_b, G, modes_g, tensor_u, modes_u, s, tensor_v, modes_v; algo=gateAlgo, kwargs...)
end
function gateSplit!(A::CuTensor{T,NA}, B::CuTensor{T,NB}, G::CuTensor{T,NG}, U::CuTensor{T,NU}, s::CuVector{S}, V::CuTensor{T,NV};
                    gateAlgo::cutensornetGateSplitAlgo_t=CUTENSORNET_GATE_SPLIT_ALGO_DIRECT, kwargs...) where {T<:Number,S<:Real,NA,NB,NG,NU,NV}
    Base.depwarn("`gateSplit!(...; gateAlgo)` is deprecated, use `gate_split!(...; algo)` instead.", :gateSplit!)
    gate_split!(A, B, G, U, s, V; algo=gateAlgo, kwargs...)
end
