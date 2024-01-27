# step one, generate optimizer info
function rehearse_contraction(tn::CuTensorNetwork, max_workspace_size::Integer, optimizer_conf::OptimizerConfig=OptimizerConfig())
    info = CuTensorNetworkContractionOptimizerInfo(tn.desc)
    # set optimizer options
    config = CuTensorNetworkContractionOptimizerConfig(optimizer_conf)
    # perform optimization
    cutensornetContractionOptimize(handle(), tn.desc, config, max_workspace_size, info)
    return info
end

function num_slices(info::CuTensorNetworkContractionOptimizerInfo)
    n = Ref{Int64}()
    cutensornetContractionOptimizerInfoGetAttribute(handle(), info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, n, sizeof(Int64))
    return n[]
end

function num_sliced_modes(info::CuTensorNetworkContractionOptimizerInfo)
    n = Ref{Int32}()
    cutensornetContractionOptimizerInfoGetAttribute(handle(), info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES, n, sizeof(Int32))
    return n[]
end

function flop_count(info::CuTensorNetworkContractionOptimizerInfo)
    n = Ref{Float64}()
    cutensornetContractionOptimizerInfoGetAttribute(handle(), info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, n, sizeof(Float64))
    return n[]
end

function pack_optimizer_data(info::CuTensorNetworkContractionOptimizerInfo)
    buf_size = Ref{Csize_t}()
    cutensornetContractionOptimizerInfoGetPackedSize(handle(), info, buf_size)
    buffer = CuVector{Cvoid}(undef, buf_size[])
    cutensornetContractionOptimizerInfoPackData(handle(), info, buffer, buf_size[])
    return buffer
end

function unpack_data_to_optimizer_info(tn::CuTensorNetwork, packed_buffer::CuVector{Cvoid})
    info = Ref{CuTensorNetworkContractionOptimizerInfo_t}()
    cutensornetCreateContractionOptimizerInfoFromPackedData(handle(), tn.desc, packed_buffer, length(packed_buffer), info)
    return CuTensorNetworkContractionOptimizerInfo(handle)
end

function unpack_data_to_optimizer_info(info::CuTensorNetworkContractionOptimizerInfo, packed_buffer::CuVector{Cvoid})
    cutensornetUpdateContractionOptimizerInfoFromPackedData(handle(), packed_buffer, length(packed_buffer), info)
    return info
end

# step 2, contract
function perform_contraction!(tn::CuTensorNetwork, info, ::NoAutoTune; prefs::AutotunePreferences=AutotunePreferences(), stream::CuStream=stream(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
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

function compute_backward_pass(plan::CuTensorNetworkContractionPlan, tn::CuTensorNetwork, output_gradient::CuTensor, gradient_tn::CuTensorNetwork; accumulate_output::Bool=false, stream::CuStream=stream())

end


function LinearAlgebra.qr!(tensor_in::CuArray{T,N}, modes_in, tensor_q::CuArray{T, Q}, modes_q, tensor_r::CuArray{T, R}, modes_r; stream::CuStream=stream(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T<:Number, N, Q, R}
    in_desc = CuTensorDescriptor(tensor_in, modes_in)
    q_desc  = CuTensorDescriptor(tensor_q, modes_q)
    r_desc  = CuTensorDescriptor(tensor_r, modes_r)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    cutensornetWorkspaceComputeQRSizes(handle(), in_desc, q_desc, r_desc, workspace_desc)
    actual_ws_size = Ref{UInt64}(0)
    cutensornetWorkspaceGetSize(handle(), workspace_desc, workspace_preference, memspace, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])
        cutensornetTensorQR(handle(), in_desc, tensor_in, q_desc, tensor_q, r_desc, tensor_r, workspace_desc, stream)
    end
    return tensor_q, tensor_r
end

LinearAlgebra.qr!(tensor_in::CuTensor{T,N}, tensor_q::CuTensor{T, Q}, tensor_r::CuTensor{T, R}; kwargs...) where {T<:Number, N, Q, R} = qr!(tensor_in.data, tensor_in.inds, tensor_q.data, tensor_q.inds, tensor_r.data, tensor_r.inds; kwargs...)

# does s need to be real-typed?
function LinearAlgebra.svd!(tensor_in::CuArray{T,N}, modes_in, tensor_u::CuArray{T, U}, modes_u, s::CuVector{S}, tensor_v::CuArray{T, V}, modes_v; svd_config::SVDConfig=SVDConfig(), stream::CuStream=stream(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T<:Number, S<:Real, N, U, V}
    in_desc = CuTensorDescriptor(tensor_in, modes_in)
    u_desc  = CuTensorDescriptor(tensor_u, modes_u)
    v_desc  = CuTensorDescriptor(tensor_v, modes_v)
    svd_info       = CuTensorSVDInfo()
    cu_svd_config  = CuTensorSVDConfig(svd_config)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{UInt64}(0)
    cutensornetWorkspaceComputeSVDSizes(handle(), in_desc, u_desc, v_desc, cu_svd_config, workspace_desc)
    cutensornetWorkspaceGetSize(handle(), workspace_desc, workspace_preference, memspace, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])
        cutensornetTensorSVD(handle(), in_desc, tensor_in, u_desc, tensor_u, s, v_desc, tensor_v, cu_svd_config, svd_info, workspace_desc, stream)
    end
    return tensor_u, s, tensor_v, svd_info
end
LinearAlgebra.svd!(tensor_in::CuTensor{T,N}, tensor_u::CuTensor{T, U}, s::CuVector{S}, tensor_v::CuTensor{T, V}; kwargs...) where {T<:Number, S<:Real, N, U, V} = svd!(tensor_in.data, tensor_in.inds, tensor_u.data, tensor_u.inds, s, tensor_v.data, tensor_v.inds; kwargs...)

function gateSplit!(A::CuArray{T, NA}, modes_a, B::CuArray{T, NB}, modes_b, G::CuArray{T, NG}, modes_g, tensor_u::CuArray{T, U}, modes_u, s::CuVector{S}, tensor_v::CuArray{T, V}, modes_v; gateAlgo::cutensornetGateSplitAlgo_t=CUTENSORNET_GATE_SPLIT_ALGO_DIRECT, svd_config::SVDConfig=SVDConfig(), stream::CuStream=stream(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T<:Number, S<:Real, NA, NB, NG, U, V}
    a_desc = CuTensorDescriptor(A, modes_a)
    b_desc = CuTensorDescriptor(B, modes_b)
    g_desc = CuTensorDescriptor(G, modes_g)
    u_desc = CuTensorDescriptor(tensor_u, modes_u)
    v_desc = CuTensorDescriptor(tensor_v, modes_v)
    svd_info       = CuTensorSVDInfo()
    cu_svd_config  = CuTensorSVDConfig(svd_config)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{UInt64}(0)
    compute_type = convert(cutensornetComputeType_t, real(T)) 
    cutensornetWorkspaceComputeGateSplitSizes(handle(), a_desc, b_desc, g_desc, u_desc, v_desc, gateAlgo, cu_svd_config, compute_type, workspace_desc)
    cutensornetWorkspaceGetSize(handle(), workspace_desc, workspace_preference, memspace, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])
        cutensornetGateSplit(handle(), a_desc, A, b_desc, B, g_desc, G, u_desc, tensor_u, s, v_desc, tensor_v, gateAlgo, cu_svd_config, compute_type, svd_info, workspace_desc, stream)
    end
    return tensor_u, s, tensor_v, svd_info
end

gateSplit!(A::CuTensor{T, NA}, B::CuTensor{T, NB}, G::CuTensor{T, NG}, U::CuTensor{T, NU}, s::CuVector{S}, V::CuTensor{T, NV}; kwargs...) where {T<:Number, S<:Real, NA, NB, NG, NU, NV} = gateSplit!(A.data, A.inds, B.data, B.inds, G.data, G.inds, U.data, U.inds, s, V.data, V.inds; kwargs...)


### High level API ###

function applyTensor!(state::CuState{TS}, tensor::CuTensor{TT}, tensor_is_unitary::Bool; adjoint::Bool=false, keep_tensor_immutable::Bool=true) where {TS, TT}
    tensor_id = Ref{Int64}()
    n_inds = length(tensor.inds)
    cutensornetStateApplyTensor(handle(), state, Int32(n_inds), Int32.(tensor.inds), pointer(tensor.data), C_NULL, keep_tensor_immutable, adjoint, tensor_is_unitary, tensor_id)
    return state, tensor_id[]
end

function updateTensor!(state::CuState{TS}, tensor_id::Int64, tensor::CuTensor{TT}, tensor_is_unitary::Bool) where {TS, TT}
    cutensornetStateUpdateTensor(handle(), state, tensor_id, pointer(tensor.data), tensor_is_unitary)
    return state
end

function appendToOperator!(operator::CuNetworkOperator{T}, coeff::ComplexF64, op_tensors::Vector{CuTensor{T, N}}; component_id::Int64=(operator.operator_count+1)) where {T, N}
    num_tensors = Int32(length(op_tensors))
    num_modes = [Int32(ndims(op)) for op in op_tensors]
    tensor_inds = [Int32.(op.inds) for op in op_tensors]
    id = Ref{Int64}(component_id)
    cutensornetNetworkOperatorAppendProduct(handle(), operator, coeff, num_tensors, num_modes, tensor_inds, C_NULL, [pointer(op.data) for op in op_tensors], id)
    operator.operator_count += 1
    return operator
end

function configure!(state::CuState, config::StateConfig)
    for (attr, jl_attr) in ((CUTENSORNET_STATE_MPS_CANONICAL_CENTER, :canonical_center),
                            (CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF, :svd_abs_cutoff),
                            (CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF, :svd_rel_cutoff),
                            (CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION, :svd_s_normalization),
                            (CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO, :svd_algo),
                            (CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS, :svd_algo_params),
                            (CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF, :svd_discarded_weight_cutoff),
                            (CUTENSORNET_STATE_NUM_HYPER_SAMPLES, :num_hyper_samples),
                           )
        attr_val = getproperty(config, jl_attr)
        attr_val_size = sizeof(attr_val)
        cutensornetStateConfigure(handle(), state, attr, [attr_val], attr_val_size)
    end
    return state
end

function compute!(state::CuState, output_tensors::Vector{CuTensor}; stream::CuStream=stream(), max_workspace_size::Csize_t=CUDA.available_memory(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{Int64}()
    cutensornetStatePrepare(handle(), state, max_workspace_size, workspace_desc, stream)
    cutensornetWorkspaceGetMemorySize(handle(), workspace_desc, workspace_preference, memspace, CUTENSORNET_WORKSPACE_SCRATCH, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])

        cutensornetStateCompute(handle(), state, workspace_desc, C_NULL, C_NULL, output_tensors, stream) 
    end
    return state, output_tensors
end

function computeMPS!(state::CuState, output_tensors::Vector{CuTensor}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN, stream::CuStream=stream(), max_workspace_size::Csize_t=CUDA.available_memory(), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE)
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{Int64}()
    cutensornetStatePrepare(handle(), state, max_workspace_size, workspace_desc, stream)
    cutensornetWorkspaceGetMemorySize(handle(), workspace_desc, workspace_preference, memspace, CUTENSORNET_WORKSPACE_SCRATCH, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, pointer(workspace), actual_ws_size[])
        cutensornetStateFinalizeMPS(handle(), state, reduce(vcat, size.(output_tensors)), reduce(vcat, strides.(output_tensors))) 
        cutensornetStateCompute(handle(), state, workspace_desc, C_NULL, C_NULL, output_tensors, stream) 
    end
    return state, output_tensors
end

function expectation(state::CuState{T}, operator::CuNetworkOperator{T}; config::ExpectationConfig=ExpectationConfig(), stream::CuStream=stream(), max_workspace_size::Csize_t=Csize_t(CUDA.available_memory()), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T}
    exp = CuStateExpectation(state, operator)
    for (attr, jl_attr) in ((CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES, :num_hyper_samples),
                           )

        attr_val = getproperty(config, jl_attr)
        attr_val_size = sizeof(attr_val)
        cutensornetExpectationConfigure(handle(), exp, attr, [attr_val], attr_val_size)
    end
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{Int64}()
    cutensornetExpectationPrepare(handle(), exp, max_workspace_size, workspace_desc, stream)
    cutensornetWorkspaceGetMemorySize(handle(), workspace_desc, workspace_preference, memspace, CUTENSORNET_WORKSPACE_SCRATCH, actual_ws_size)
    # ws must be 256 byte aligned!
    exp_val    = Ref{T}(zero(T))
    state_norm = Ref{T}(zero(T))
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSetMemory(handle(), workspace_desc, memspace, CUTENSORNET_WORKSPACE_SCRATCH, pointer(workspace), actual_ws_size[])
        cutensornetExpectationCompute(handle(), exp, workspace_desc, exp_val, state_norm, stream) 
    end
    return exp_val[], state_norm[]
end

function compute_marginal!(state::CuState{T}, marginal_modes::Vector{Int}, projected_modes::Vector{Int}, projected_mode_values::Vector{Int}, marginal_tensor::CuTensor{T}; config::MarginalConfig=MarginalConfig(), stream::CuStream=stream(), max_workspace_size::Csize_t=Csize_t(CUDA.available_memory()), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T}
    marg = CuStateMarginal(state, Int32.(marginal_modes), Int32.(projected_modes))
    for (attr, jl_attr) in ((CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES, :num_hyper_samples),
                           )

        attr_val = getproperty(config, jl_attr)
        attr_val_size = sizeof(attr_val)
        cutensornetMarginalConfigure(handle(), marg, attr, [attr_val], attr_val_size)
    end
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{Int64}()
    cutensornetMarginalPrepare(handle(), marg, max_workspace_size, workspace_desc, stream)
    cutensornetWorkspaceGetMemorySize(handle(), workspace_desc, workspace_preference, memspace, CUTENSORNET_WORKSPACE_SCRATCH, actual_ws_size)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSetMemory(handle(), workspace_desc, memspace, CUTENSORNET_WORKSPACE_SCRATCH, pointer(workspace), actual_ws_size[])
        cutensornetMarginalCompute(handle(), marg, projected_mode_values, workspace_desc, pointer(marginal_tensor.data), stream) 
    end
    return marginal_tensor
end

function sample!(state::CuState{T}, modes_to_sample::Vector{Int}, num_shots::Int; config::SamplerConfig=SamplerConfig(), stream::CuStream=stream(), max_workspace_size::Csize_t=Csize_t(CUDA.available_memory()), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T}
    sampler = CuStateSampler(state, modes_to_sample)
    for (attr, jl_attr) in ((CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES, :num_hyper_samples),
                           )

        attr_val = getproperty(config, jl_attr)
        attr_val_size = sizeof(attr_val)
        cutensornetSamplerConfigure(handle(), sampler, attr, [attr_val], attr_val_size)
    end
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{Int64}()
    cutensornetSamplerPrepare(handle(), sampler, max_workspace_size, workspace_desc, stream)
    cutensornetWorkspaceGetMemorySize(handle(), workspace_desc, workspace_preference, memspace, CUTENSORNET_WORKSPACE_SCRATCH, actual_ws_size)
    samples = Vector{Int64}(undef, length(modes_to_sample)*num_shots)
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSetMemory(handle(), workspace_desc, memspace, CUTENSORNET_WORKSPACE_SCRATCH, pointer(workspace), actual_ws_size[])
        cutensornetSamplerSample(handle(), sampler, Int64(num_shots), workspace_desc, samples, stream)
    end
    return reshape(samples, (length(modes_to_sample), num_shots))
end

function amplitudes!(state::CuState{T}, projected_modes::Vector{Int}, projected_mode_values::Vector{Int}, amplitudes_tensor::CuTensor{T}; config::AccessorConfig=AccessorConfig(), stream::CuStream=stream(), max_workspace_size::Csize_t=Csize_t(CUDA.available_memory()), workspace_preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, memspace::cutensornetMemspace_t=CUTENSORNET_MEMSPACE_DEVICE) where {T}
    accessor = CuStateAccessor(state, Int32.(projected_modes))
    for (attr, jl_attr) in ((CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES, :num_hyper_samples),
                           )

        attr_val = getproperty(config, jl_attr)
        attr_val_size = sizeof(attr_val)
        cutensornetAccessorConfigure(handle(), accessor, attr, [attr_val], attr_val_size)
    end
    workspace_desc = CuTensorNetworkWorkspaceDescriptor()
    actual_ws_size = Ref{Int64}()
    cutensornetAccessorPrepare(handle(), accessor, max_workspace_size, workspace_desc, stream)
    cutensornetWorkspaceGetMemorySize(handle(), workspace_desc, workspace_preference, memspace, CUTENSORNET_WORKSPACE_SCRATCH, actual_ws_size)
    state_norm = Ref{T}()
    with_workspace(actual_ws_size[]) do workspace
        cutensornetWorkspaceSetMemory(handle(), workspace_desc, memspace, CUTENSORNET_WORKSPACE_SCRATCH, pointer(workspace), actual_ws_size[])
        cutensornetAccessorCompute(handle(), accessor, Int64.(projected_mode_values), workspace_desc, pointer(amplitudes_tensor.data), state_norm, stream) 
    end
    synchronize()
    return amplitudes_tensor, state_norm[]
end
