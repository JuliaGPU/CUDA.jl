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
    elseif T == CUDACore.BFloat16
        return CUTENSORNET_COMPUTE_16BF
    else
        throw(ArgumentError("cuTensorNet type equivalent for compute type $T does not exist!"))
    end
end

function Base.Type(T::cutensornetComputeType_t)
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
    elseif T == CUTENSORNET_COMPUTE_16BF
        return CUDACore.BFloat16
    else
        throw(ArgumentError("Julia type equivalent for compute type $T does not exist!"))
    end
end

mutable struct CuTensorDescriptor{T}
    handle::cutensornetTensorDescriptor_t
    function CuTensorDescriptor{T}(extents, strides, modes) where {T}
        desc_ref = Ref{cutensornetTensorDescriptor_t}()
        cutensornetCreateTensorDescriptor(handle(), Int32(length(extents)), convert(Vector{Int64},  collect(extents)),
                                          convert(Vector{Int64}, collect(strides)), convert(Vector{Int32}, collect(modes)),
                                          T, desc_ref)
        obj = new{T}(desc_ref[])
        resource_finalizer(cutensornetDestroyTensorDescriptor, obj)
        return obj
    end
end
CuTensorDescriptor(T::DataType, extents, strides, modes) = CuTensorDescriptor{T}(extents, strides, modes)
CuTensorDescriptor(a::AbstractArray{T, N}, modes) where {T, N} = CuTensorDescriptor{T}(size(a), strides(a), modes)
CuTensorDescriptor(t::CuTensor{T, N}) where {T, N} = CuTensorDescriptor{T}(size(t), stride(t), t.inds)

function Base.ndims(desc::CuTensorDescriptor)
    numModes = Ref{Int32}(C_NULL)
    cutensornetGetTensorDetails(handle(), desc, numModes, C_NULL, C_NULL, C_NULL, C_NULL)
    return numModes[]
end

function Base.size(desc::CuTensorDescriptor)
    numModes = Ref{Int32}(C_NULL)
    extents  = Vector{Int64}(undef, ndims(desc))
    cutensornetGetTensorDetails(handle(), desc, numModes, C_NULL, C_NULL, extents, C_NULL)
    return tuple(extents...)
end

function Base.strides(desc::CuTensorDescriptor)
    numModes = Ref{Int32}(C_NULL)
    strides  = Vector{Int64}(undef, ndims(desc))
    cutensornetGetTensorDetails(handle(), desc, numModes, C_NULL, C_NULL, C_NULL, strides)
    return tuple(strides...)
end

Base.unsafe_convert(::Type{cutensornetTensorDescriptor_t}, desc::CuTensorDescriptor) = desc.handle

mutable struct CuTensorSVDInfo
    handle::cutensornetTensorSVDInfo_t
    function CuTensorSVDInfo()
        info_ref = Ref{cutensornetTensorSVDInfo_t}()
        cutensornetCreateTensorSVDInfo(handle(), info_ref)
        obj = new(info_ref[])
        resource_finalizer(cutensornetDestroyTensorSVDInfo, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetTensorSVDInfo_t}, info::CuTensorSVDInfo) = info.handle
function full_extent(info::CuTensorSVDInfo)
    extent = Ref{Int64}()
    cutensornetTensorSVDInfoGetAttribute(handle(), info, CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT, extent, sizeof(Int64))
    return extent[]
end

function reduced_extent(info::CuTensorSVDInfo)
    extent = Ref{Int64}()
    cutensornetTensorSVDInfoGetAttribute(handle(), info, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, extent, sizeof(Int64))
    return extent[]
end

function discarded_weight(info::CuTensorSVDInfo)
    weight = Ref{Float64}()
    cutensornetTensorSVDInfoGetAttribute(handle(), info, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, weight, sizeof(Float64))
    return weight[]
end

mutable struct CuTensorNetworkContractionOptimizerInfo
    handle::cutensornetContractionOptimizerInfo_t
    function CuTensorNetworkContractionOptimizerInfo(info::cutensornetContractionOptimizerInfo_t)
        obj = new(info)
        resource_finalizer(cutensornetDestroyContractionOptimizerInfo, obj)
        return obj
    end
end
function CuTensorNetworkContractionOptimizerInfo(net_desc)
    desc_ref = Ref{cutensornetContractionOptimizerInfo_t}()
    cutensornetCreateContractionOptimizerInfo(handle(), net_desc, desc_ref)
    return CuTensorNetworkContractionOptimizerInfo(desc_ref[])
end

Base.unsafe_convert(::Type{cutensornetContractionOptimizerInfo_t}, desc::CuTensorNetworkContractionOptimizerInfo) = desc.handle

mutable struct CuTensorNetworkWorkspaceDescriptor
    handle::cutensornetWorkspaceDescriptor_t
    function CuTensorNetworkWorkspaceDescriptor()
        desc_ref = Ref{cutensornetWorkspaceDescriptor_t}()
        cutensornetCreateWorkspaceDescriptor(handle(), desc_ref)
        obj = new(desc_ref[])
        resource_finalizer(cutensornetDestroyWorkspaceDescriptor, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cutensornetWorkspaceDescriptor_t}, desc::CuTensorNetworkWorkspaceDescriptor) = desc.handle

Base.@kwdef struct OptimizerConfig
    num_graph_partitions::Int32=8
    graph_cutoff_size::Int32=8
    graph_algorithm::cutensornetGraphAlgo_t=CUTENSORNET_GRAPH_ALGO_KWAY
    graph_imbalance_factor::Int32=200
    graph_num_iterations::Int32=60
    graph_num_cuts::Int32=10
    reconfig_num_iterations::Int32=500
    reconfig_num_leaves::Int32=8
    slicer_disable_slicing::Int32=0
    slicer_memory_model::cutensornetMemoryModel_t=CUTENSORNET_MEMORY_MODEL_CUTENSOR
    slicer_memory_factor::Int32=80
    slicer_min_slices::Int32=1
    slicer_slice_factor::Int32=2
    hyper_num_samples::Int32=0
    simplification_disable_dr::Int32=0
    hyper_num_threads::Int32=Threads.nthreads()
    seed::Int32=0
    cost_function_objective::cutensornetOptimizerCost_t=CUTENSORNET_OPTIMIZER_COST_FLOPS
    gpu_arch::Int32=0
end

mutable struct CuTensorNetworkContractionOptimizerConfig
    handle::cutensornetContractionOptimizerConfig_t
    function CuTensorNetworkContractionOptimizerConfig(prefs::OptimizerConfig)
        desc_ref = Ref{cutensornetContractionOptimizerConfig_t}()
        cutensornetCreateContractionOptimizerConfig(handle(), desc_ref)
        obj = new(desc_ref[])
        resource_finalizer(cutensornetDestroyContractionOptimizerConfig, obj)
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
                        :hyper_num_threads=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS,
                        :seed=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED,
                        :cost_function_objective=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE,
                        :gpu_arch=>CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GPU_ARCH,
                    )
            attr_buf = Ref(Base.getproperty(prefs, attr[1]))
            cutensornetContractionOptimizerConfigSetAttribute(handle(), desc_ref[], attr[2], attr_buf, sizeof(attr_buf))
        end
        return obj
    end
end

Base.unsafe_convert(::Type{cutensornetContractionOptimizerConfig_t}, desc::CuTensorNetworkContractionOptimizerConfig) = desc.handle

Base.@kwdef struct SVDConfig
    abs_cutoff::Float64=0.0
    rel_cutoff::Float64=0.0
    s_normalization::cutensornetTensorSVDNormalization_t=CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
    s_partition::cutensornetTensorSVDPartition_t=CUTENSORNET_TENSOR_SVD_PARTITION_NONE
end

mutable struct CuTensorSVDConfig
    handle::cutensornetTensorSVDConfig_t
    function CuTensorSVDConfig()
        desc_ref = Ref{cutensornetTensorSVDConfig_t}()
        cutensornetCreateTensorSVDConfig(handle(), desc_ref)
        obj = new(desc_ref[])
        resource_finalizer(cutensornetDestroyTensorSVDConfig, obj)
        return obj
    end
    function CuTensorSVDConfig(prefs::SVDConfig)
        desc_ref = Ref{cutensornetTensorSVDConfig_t}()
        cutensornetCreateTensorSVDConfig(handle(), desc_ref)
        obj = new(desc_ref[])
        resource_finalizer(cutensornetDestroyTensorSVDConfig, obj)
        # apply preference options
        for attr in (
            :abs_cutoff=>CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
            :rel_cutoff=>CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
            :s_normalization=>CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION,
            :s_partition=>CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION,
        )
            attr_buf = Ref(Base.getproperty(prefs, attr[1]))
            cutensornetTensorSVDConfigSetAttribute(handle(), desc_ref[], attr[2], attr_buf, sizeof(attr_buf))
        end
        return obj
    end
end
function abs_cutoff(conf::CuTensorSVDConfig)
    attr_buf = Ref{Float64}(0.0)
    cutensornetTensorSVDConfigGetAttribute(handle(), conf, CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF, attr_buf, sizeof(attr_buf))
    return attr_buf[]
end
function rel_cutoff(conf::CuTensorSVDConfig)
    attr_buf = Ref{Float64}(0.0)
    cutensornetTensorSVDConfigGetAttribute(handle(), conf, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, attr_buf, sizeof(attr_buf))
    return attr_buf[]
end
function normalization(conf::CuTensorSVDConfig)
    attr_buf = Ref{cutensornetTensorSVDNormalization_t}()
    cutensornetTensorSVDConfigGetAttribute(handle(), conf, CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION, attr_buf, sizeof(attr_buf))
    return attr_buf[]
end
Base.unsafe_convert(::Type{cutensornetTensorSVDConfig_t}, desc::CuTensorSVDConfig) = desc.handle

mutable struct CuTensorNetworkSliceGroup
    handle::cutensornetSliceGroup_t
    function CuTensorNetworkSliceGroup(sliceStart::Int64, sliceStop::Int64, sliceStep::Int64)
        group_ref = Ref{cutensornetSliceGroup_t}()
        cutensornetCreateSliceGroupFromIDRange(handle(), sliceStart, sliceStop, sliceStep, group_ref)
        obj = new(group_ref[])
        resource_finalizer(cutensornetDestroySliceGroup, obj)
        return obj
    end
    function CuTensorNetworkSliceGroup(slices::Vector{Int64})
        group_ref = Ref{cutensornetSliceGroup_t}()
        cutensornetCreateSliceGroupFromIDs(handle(), pointer(slices), pointer(slices, length(slices)), group_ref)
        obj = new(group_ref[])
        resource_finalizer(cutensornetDestroySliceGroup, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetSliceGroup_t}, prefs::CuTensorNetworkSliceGroup)   = prefs.handle
