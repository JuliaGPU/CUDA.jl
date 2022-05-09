# step one, generate optimizer info
function rehearse_contraction(tn::CuTensorNetwork, max_workspace_size::Integer, optimizer_conf::OptimizerConfig=OptimizerConfig())
    info = CuTensorNetworkContractionOptimizerInfo(tn.desc)
    # set optimizer options
    config = CuTensorNetworkContractionOptimizerConfig(optimizer_conf)
    # perform optimization
    cutensornetContractionOptimize(handle(), tn.desc, config, max_workspace_size, info)
    plan = CuTensorNetworkContractionPlan(tn.desc, info, max_workspace_size)
    return info, plan
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
function perform_contraction!(tn::CuTensorNetwork, info, plan, workspace_size::Integer, ::NoAutoTune; prefs::AutotunePreferences=AutotunePreferences(), stream::CuStream=stream())
    with_workspace(workspace_size) do workspace
        for slice_ix in 0:num_slices(info)-1
            cutensornetContraction(handle(), plan, pointer.(tn.input_arrs),
                                   pointer(tn.output_arr), workspace, workspace_size,
                                   slice_ix, stream)
        end
    end
    return tn
end

function perform_contraction!(tn::CuTensorNetwork, info, plan, workspace_size::Integer, ::AutoTune; prefs::AutotunePreferences=AutotunePreferences(), stream::CuStream=stream())
    ctn_prefs = CuTensorNetworkAutotunePreference(prefs)
    with_workspace(workspace_size) do workspace
        cutensornetContractionAutotune(handle(), plan, pointer.(tn.input_arrs),
                                       pointer(tn.output_arr), workspace, workspace_size,
                                       ctn_prefs, stream)
        for slice_ix in 0:num_slices(info)-1
            cutensornetContraction(handle(), plan, pointer.(tn.input_arrs),
                                   pointer(tn.output_arr), workspace, workspace_size,
                                   slice_ix, stream)
        end
    end
    return tn
end

