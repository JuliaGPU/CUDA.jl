# cuTENSORNET types

## cutensornet compute type

function Base.convert(::Type{cutensornetComputeType_t}, T::DataType)
    if T == Float16
        return CUTENSORNET_COMPUTE_16F
    elseif T == Float32 
        return CUTENSORNET_COMPUTE_32F
    elseif T == Float64
        return CUTENSORNET_COMPUTE_64F
    elseif T == UInt8 
        return CUTENSORNET_COMPUTE_8U
    elseif T == Int8 
        return CUTENSORNET_COMPUTE_8I
    elseif T == UInt32 
        return CUTENSORNET_COMPUTE_32U
    elseif T == Int32 
        return CUTENSORNET_COMPUTE_32I
    else
        throw(ArgumentError("CUTENSORNET type equivalent for compute type $T does not exist!"))
    end
end

function Base.convert(::Type{Type}, T::cutensornetComputeType_t)
    if T == CUTENSORNET_COMPUTE_16F
        return Float16 
    elseif T == CUTENSORNET_COMPUTE_32F
        return Float32
    elseif T == CUTENSORNET_COMPUTE_64F
        return Float64
    elseif T == CUTENSORNET_COMPUTE_8U
        return UInt8 
    elseif T == CUTENSORNET_COMPUTE_32U
        return UInt32
    elseif T == CUTENSORNET_COMPUTE_8I
        return Int8 
    elseif T == CUTENSORNET_COMPUTE_32I
        return Int32
    else
        throw(ArgumentError("Julia type equivalent for compute type $T does not exist!"))
    end
end

mutable struct CuTensorNetworkDescriptor
    handle::cutensornetNetworkDescriptor_t
    function CuTensorNetworkDescriptor(numInputs::Int32, numModesIn::Vector{Int32}, extentsIn::Vector{Vector{Int64}},
                                       stridesIn::Vector{Vector{Int64}}, modesIn::Vector{Vector{Int32}},
                                       alignmentRequirementsIn::Vector{Int32}, numModesOut::Int32,
                                       extentsOut::Vector{Int64}, stridesOut::Vector{Int64}, modesOut::Vector{Int32},
                                       alignmentRequirementsOut::Int32, dataType::Type, computeType::Type)
        desc_ref = Ref{cutensornetNetworkDescriptor_t}()
        cutensornetCreateNetworkDescriptor(handle(), numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentRequirementsIn, numModesOut, extentsOut, stridesOut, modesOut, alignmentRequirementsOut, dataType, computeType, desc_ref)
        obj = new(desc_ref[])
        finalizer(cutensornetDestroyNetworkDescriptor, obj)
        obj
    end
end
Base.unsafe_convert(::Type{cutensornetNetworkDescriptor_t}, desc::CuTensorNetworkDescriptor) = desc.handle

function compute_type(T::DataType)
    if T == Float16
        return Float32
    elseif T == Float32
        return Float16
    elseif T == Float64
        return Float64
    end 
end

mutable struct CuTensorNetwork{T}
    desc::CuTensorNetworkDescriptor
    input_modes::Vector{Vector{Int32}}
    input_extents::Vector{Vector{Int32}}
    input_strides::Vector{Vector{Int32}}
    input_alignment_reqs::Vector{Int32}
    input_arrs::Vector{CuArray{T}}
    output_modes::Vector{Int32}
    output_extents::Vector{Int32}
    output_strides::Vector{Int32}
    output_alignment_reqs::Int32
    output_arr::CuArray{T}
end
function CuTensorNetwork(T::DataType, input_modes, input_extents, input_strides, input_alignment_reqs, output_modes, output_extents, output_strides, output_alignment_reqs)
    desc = CuTensorNetworkDescriptor(Int32(length(input_modes)), Int32.(length.(input_modes)), input_extents, input_strides, input_modes, input_alignment_reqs, Int32(length(output_modes)), output_extents, output_strides, output_modes, output_alignment_reqs, T, compute_type(real(T)))
    
    return CuTensorNetwork{T}(desc, input_modes, input_extents, input_strides, input_alignment_reqs, Vector{CuArray{T}}(undef, 0), output_modes, output_extents, output_strides, output_alignment_reqs, CUDA.zeros(T, 0))
end

mutable struct CuTensorNetworkContractionOptimizerInfo
    handle::cutensornetContractionOptimizerInfo_t
    function CuTensorNetworkContractionOptimizerInfo(net_desc::CuTensorNetworkDescriptor)
        desc_ref = Ref{cutensornetContractionOptimizerInfo_t}()
        cutensornetCreateContractionOptimizerInfo(handle(), net_desc, desc_ref)
        obj = new(desc_ref[])
        finalizer(cutensornetDestroyContractionOptimizerInfo, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cutensornetContractionOptimizerInfo_t}, desc::CuTensorNetworkContractionOptimizerInfo) = desc.handle

mutable struct CuTensorNetworkContractionPlan
    handle::cutensornetContractionPlan_t
    function CuTensorNetworkContractionPlan(net_desc::CuTensorNetworkDescriptor, info::CuTensorNetworkContractionOptimizerInfo, ws_size)
        desc_ref = Ref{cutensornetContractionPlan_t}()
        cutensornetCreateContractionPlan(handle(), net_desc, info, ws_size, desc_ref)
        obj = new(desc_ref[])
        finalizer(cutensornetDestroyContractionPlan, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cutensornetContractionPlan_t}, desc::CuTensorNetworkContractionPlan) = desc.handle

abstract type UseAutotuning end
struct NoAutoTune <: UseAutotuning end
struct AutoTune   <: UseAutotuning end

Base.@kwdef struct OptimizerConfig
    num_graph_partitions::Int32=8
    graph_cutoff_size::Int32=8
    graph_algorithm::cutensornetGraphAlgo_t=CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM_KWAY
    graph_imbalance_factor::Int32=200
    graph_num_iterations::Int32=60
    graph_num_cuts::Int32=10
    reconfig_num_iterations::Int32=500
    reconfig_num_leaves::Int32=500
    slicer_disable_slicing::Int32=0
    slicer_memory_model::cutensornetMemoryModel_t=CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL_CUTENSOR
    slicer_memory_factor::Int32=2
    slicer_min_slices::Int32=1
    slicer_slice_factor::Int32=2
    hyper_num_samples::Int32=0
    simplification_disable_dr::Int32=0
    seed::Int32=0
end

mutable struct CuTensorNetworkContractionOptimizerConfig
    handle::cutensornetContractionOptimizerConfig_t
    function CuTensorNetworkContractionOptimizerConfig(prefs::OptimizerConfig)
        desc_ref = Ref{cutensornetContractionOptimizerConfig_t}()
        cutensornetCreateContractionOptimizerConfig(handle(), desc_ref)
        obj = new(desc_ref[])
        finalizer(cutensornetDestroyContractionOptimizerConfig, obj)
        # apply preference options
        for attr in (
                        :num_graph_partitions=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS,
                        :graph_cutoff_size=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE,
                        :graph_algorithm=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM,
                        :graph_imbalance_factor=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR,
                        :graph_num_iterations=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS,
                        :graph_num_cuts=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS,
                        :reconfig_num_iterations=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS,
                        :reconfig_num_leaves=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES,
                        :slicer_disable_slicing=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING,
                        :slicer_memory_model=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL,
                        :slicer_memory_factor=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR,
                        :slicer_min_slices=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES,
                        :slicer_slice_factor=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR,
                        :hyper_num_samples=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
                        :simplification_disable_dr=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR,
                        :seed=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED,
                    )
            attr_buf = Ref(Base.getproperty(prefs, attr[1]))
            cutensornetContractionOptimizerConfigSetAttribute(handle(), desc_ref[], attr[2], attr_buf, sizeof(attr_buf))
        end
        obj
    end
end

Base.unsafe_convert(::Type{cutensornetContractionOptimizerConfig_t}, desc::CuTensorNetworkContractionOptimizerConfig) = desc.handle

Base.@kwdef struct AutotunePreferences 
    max_iterations::Int32=3
end

mutable struct CuTensorNetworkAutotunePreference
    handle::cutensornetContractionAutotunePreference_t
    function CuTensorNetworkAutotunePreference(prefs::AutotunePreferences)
        pref_ref = Ref{cutensornetContractionAutotunePreference_t}()
        cutensornetCreateContractionAutotunePreference(handle(), pref_ref)
        obj = new(pref_ref[])
        finalizer(cutensornetDestroyContractionAutotunePreference, obj)
        # apply preference options
        for attr in (:max_iterations=>CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,)
            attr_buf = Ref(Base.getproperty(prefs, attr[1]))
            cutensornetContractionAutotunePreferenceSetAttribute(handle(), pref_ref[], attr[2], attr_buf, sizeof(attr_buf))
        end
        obj
    end
end
Base.unsafe_convert(::Type{cutensornetContractionAutotunePreference_t}, prefs::CuTensorNetworkAutotunePreference)   = prefs.handle
Base.unsafe_convert(::Type{CuTensorNetworkAutotunePreference}, prefs::AutotunePreferences) = cutensornetContractionAutotunePreference(prefs)
