function initialize()
    params = Ref(NVPW_InitializeHost_Params(NVPW_InitializeHost_Params_STRUCT_SIZE, C_NULL))
    NVPW_InitializeHost(params)
end

function supported_chips()
    params = Ref(NVPW_GetSupportedChipNames_Params(
        NVPW_GetSupportedChipNames_Params_STRUCT_SIZE,
        C_NULL, C_NULL, 0))
    NVPW_GetSupportedChipNames(params)

    names = String[]
    for i in params[].numChipNames
        push!(names, Base.unsafe_string(Base.unsafe_load(params[].ppChipNames, i)))
    end
    return names
end

function scratch_buffer(chipName, counter_availability)
    GC.@preserve chipName counter_availability begin
        params = Ref(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params(
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE,
            C_NULL, pointer(chipName), pointer(counter_availability), 0
        ))
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(params)
        sz = params[].scratchBufferSize
    end
    return Vector{UInt8}(undef, sz)
end

abstract type MetricsEvaluator end

mutable struct CUDAMetricsEvaluator <: MetricsEvaluator
    handle::Ptr{NVPW_MetricsEvaluator}
    scratch::Vector{UInt8}
    availability::Vector{UInt8}
    chip::String

    function CUDAMetricsEvaluator(chip, availability)
        scratch = scratch_buffer(chip, availability)

        GC.@preserve chip availability scratch begin
            params = Ref(NVPW_CUDA_MetricsEvaluator_Initialize_Params(
                NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE,
                C_NULL, pointer(scratch), length(scratch), pointer(chip),
                pointer(availability), C_NULL, 0, C_NULL))
            
            NVPW_CUDA_MetricsEvaluator_Initialize(params)
            this =  new(params[].pMetricsEvaluator, scratch, availability, chip)
        end
        finalizer(destroy, this)
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{NVPW_MetricsEvaluator}}, me::CUDAMetricsEvaluator) = me.handle


function destroy(me::MetricsEvaluator)
    GC.@preserve me begin
        params = Ref(NVPW_MetricsEvaluator_Destroy_Params(
            NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me)
        ))
        NVPW_MetricsEvaluator_Destroy(params)
    end
    return nothing
end

struct MetricsIterator
    me::MetricsEvaluator
    type::NVPW_MetricType
    names::Ptr{Cchar}
    indices::Ptr{Csize_t}
    numMetrics::Csize_t

    function MetricsIterator(me, type)
        GC.@preserve me begin
            params = Ref(NVPW_MetricsEvaluator_GetMetricNames_Params(
                NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), type, C_NULL, C_NULL, 0))
            NVPW_MetricsEvaluator_GetMetricNames(params)

            names = Ptr{Cchar}(params[].pMetricNames)
            indices = params[].pMetricNameBeginIndices

            return new(me, type, names, indices, params[].numMetrics)
        end
    end
end

Base.length(metrics::MetricsIterator) = metrics.numMetrics
Base.eltype(::MetricsIterator) = String

function Base.iterate(metrics::MetricsIterator, state=1)
    if state <= metrics.numMetrics
        name = unsafe_string(metrics.names + unsafe_load(metrics.indices, state))
        return (name, state+1)
    else
        return nothing
    end
end

function list_metrics(me::MetricsEvaluator)
    for i in 0:(NVPW_METRIC_TYPE__COUNT-1)
        type = NVPW_MetricType(i)

        for metric in MetricsIterator(me, type)
            @show metric
        end
    end
end

function submetrics(me::MetricsEvaluator, type)
    GC.@preserve me begin
        params = Ref(NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params(
            NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), type, C_NULL, 0))
        NVPW_MetricsEvaluator_GetSupportedSubmetrics(params)
        unsafe_wrap(Array, params[].pSupportedSubmetrics, params[].numSupportedSubmetrics)
    end
end

# TODO rollup to string
# TODO submetric to string

# function submetric(m)
#     if m == NVPW_SUBMETRIC_PEAK_SUSTAINED
#         return ".peak_sustained"
#     elseif 

struct Metric
    me::MetricsEvaluator
    type::NVPW_MetricType
    index::Csize_t

    function Metric(me::MetricsEvaluator, name)
        GC.@preserve me name begin
            params = Ref(NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params(
                NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), pointer(name), 0, 0))
            NVPW_MetricsEvaluator_GetMetricTypeAndIndex(params)
            return new(me, NVPW_MetricType(params[].metricType), params[].metricIndex)
        end
    end
end

struct HWUnit
    me::MetricsEvaluator
    hwUnit::UInt32
end

function Base.string(u::HWUnit)
    GC.@preserve u begin
        params = Ref(NVPW_MetricsEvaluator_HwUnitToString_Params(
            NVPW_MetricsEvaluator_HwUnitToString_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, u.me), u.hwUnit,
            C_NULL))
        NVPW_MetricsEvaluator_HwUnitToString(params)
        return unsafe_string(params[].pHwUnitName)
    end
end

function properties(m::Metric)
    if m.type == NVPW_METRIC_TYPE_COUNTER
        GC.@preserve m begin
            params = Ref(NVPW_MetricsEvaluator_GetCounterProperties_Params(
                NVPW_MetricsEvaluator_GetCounterProperties_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, m.me), m.index,
                C_NULL, 0))
            NVPW_MetricsEvaluator_GetCounterProperties(params)
            description = unsafe_string(params[].pDescription)
            hwUnit = params[].hwUnit
            return (; description, unit=HWUnit(m.me, hwUnit))
        end
    elseif m.type == NVPW_METRIC_TYPE_RATIO
        GC.@preserve m begin
            params = Ref(NVPW_MetricsEvaluator_GetRatioMetricProperties_Params(
                NVPW_MetricsEvaluator_GetRatioMetricProperties_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, m.me), m.index,
                C_NULL, 0))
            NVPW_MetricsEvaluator_GetRatioMetricProperties(params)
            description = unsafe_string(params[].pDescription)
            hwUnit = params[].hwUnit
            return (; description, unit=HWUnit(m.me, hwUnit))
        end
    else
        error("Not implemented for $(m.type)")
    end
end

struct MetricEvalRequest
    me::MetricsEvaluator
    data::NVPW_MetricEvalRequest

    function MetricEvalRequest(me::MetricsEvaluator, name)
        eval_request = Ref{NVPW_MetricEvalRequest}()
        GC.@preserve me name eval_request begin
            params = Ref(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params(
                NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), pointer(name),
                Base.unsafe_convert(Ptr{NVPW_MetricEvalRequest}, eval_request), NVPW_MetricEvalRequest_STRUCT_SIZE))
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(params)
            return new(me, eval_request[])
        end
    end
end
