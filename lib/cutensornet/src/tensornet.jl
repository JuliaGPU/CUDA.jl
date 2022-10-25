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
        for slice_ix in 0:num_slices(info)-1
            cutensornetContraction(handle(), plan, input_ptrs, output_ptr, workspace_desc, slice_ix, stream)
        end
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
        for slice_ix in 0:num_slices(info)-1
            cutensornetContraction(handle(), plan, input_ptrs, output_ptr, workspace_desc, slice_ix, stream)
        end
        cutensornetWorkspaceSet(handle(), workspace_desc, memspace, C_NULL, 0)
    end
    return tn
end

