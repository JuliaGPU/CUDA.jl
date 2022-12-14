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
