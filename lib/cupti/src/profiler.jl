#
# range profiler API
#
# Collects hardware metrics per kernel launch: in auto-range mode, CUPTI serializes all
# kernels in the profiled context and replays each launch as often as the requested
# metrics need passes, saving and restoring device memory in between.

"""
    CUPTI.profiler_params(T, lastfield; fields...)

Construct a zero-initialized profiler API parameter struct of type `T`, setting the given
`fields`. `structSize` covers the fields up to and including `lastfield`, which should be
the last field the oldest supported CUPTI knows about: CUPTI appends fields to these
structs, and rejects sizes it doesn't recognize.
"""
function profiler_params(::Type{T}, lastfield::Symbol; fields...) where {T}
    ref = Ref{T}()
    GC.@preserve ref begin
        ptr = Base.unsafe_convert(Ptr{T}, ref)
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), ptr, 0, sizeof(T))
        unsafe_store!(Ptr{Csize_t}(ptr), struct_size(T, lastfield))
        for (name, value) in pairs(fields)
            i = Base.fieldindex(T, name)
            FT = fieldtype(T, i)
            # convert like `ccall` would, e.g. to get the handle of a CuContext. callers
            # need to preserve any memory the field points to.
            unsafe_store!(Ptr{FT}(ptr + fieldoffset(T, i)),
                          Base.unsafe_convert(FT, Base.cconvert(FT, value)))
        end
    end
    return ref
end

"""
    CUPTI.has_range_profiler() -> Bool

Whether the loaded CUPTI library provides the range profiler (CUDA 12.6 Update 2 and
later). The API version doesn't change between update releases, so look for the symbol.
"""
function has_range_profiler()
    isdefined(@__MODULE__, :libcupti) || return false
    handle = Libdl.dlopen(libcupti)
    Libdl.dlsym(handle, :cuptiRangeProfilerEnable; throw_error=false) !== nothing
end

"""
    CUPTI.chip_name(dev::CuDevice) -> String

The name of the chip of `dev`, as used to configure the profiler (e.g. `"GB203"`).
"""
function chip_name(dev::CuDevice)
    params = profiler_params(CUpti_Device_GetChipName_Params, :pChipName;
                             deviceIndex=deviceid(dev))
    cuptiDeviceGetChipName(params)
    unsafe_string(params[].pChipName)
end

"""
    CUPTI.check_profiler_support(dev::CuDevice)

Throw an error explaining why `dev` doesn't support the range profiler, if it doesn't.
"""
function check_profiler_support(dev::CuDevice)
    params = profiler_params(CUpti_Profiler_DeviceSupported_Params, :api;
                             cuDevice=CUdevice(deviceid(dev)),
                             api=CUPTI_PROFILER_RANGE_PROFILING)
    res = unchecked_cuptiProfilerDeviceSupported(params)
    # MIG devices fail the query itself
    res == CUPTI_ERROR_INVALID_PARAMETER &&
        error("Hardware metrics are not supported on this device (MIG instances are not supported).")
    res == CUPTI_SUCCESS || throw_api_error(res)

    p = params[]
    p.isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED && return
    reasons = String[]
    unsupported(level) = level in (CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED,
                                   CUPTI_PROFILER_CONFIGURATION_DISABLED)
    unsupported(p.architecture) && push!(reasons, "the GPU architecture is not supported")
    unsupported(p.sli) && push!(reasons, "SLI is enabled")
    unsupported(p.vGpu) && push!(reasons, "virtual GPUs are not supported, or profiling is disabled by the hypervisor")
    unsupported(p.confidentialCompute) && push!(reasons, "confidential computing is enabled")
    unsupported(p.cmp) && push!(reasons, "CMP devices are not supported")
    unsupported(p.wsl) && push!(reasons, "this version of WSL is not supported")
    isempty(reasons) && push!(reasons, "unknown reason")
    error("Hardware metrics are not supported on this device: $(join(reasons, ", ")).")
end

function counter_availability(ctx::CuContext)
    params = profiler_params(CUpti_Profiler_GetCounterAvailability_Params,
                             :pCounterAvailabilityImage; ctx)
    cuptiProfilerGetCounterAvailability(params)
    image = Vector{UInt8}(undef, params[].counterAvailabilityImageSize)
    params = profiler_params(CUpti_Profiler_GetCounterAvailability_Params,
                             :pCounterAvailabilityImage; ctx,
                             counterAvailabilityImageSize=length(image),
                             pCounterAvailabilityImage=pointer(image))
    GC.@preserve image cuptiProfilerGetCounterAvailability(params)
    return image
end


## profiler host: metric configuration and evaluation

"""
    CUPTI.ProfilerHost(f, ctx::CuContext)

Create a profiler host object for the device of `ctx`, which configures and evaluates
range profiler metrics, and pass it to `f`.
"""
mutable struct ProfilerHost
    handle::Ptr{CUpti_Profiler_Host_Object}
    ctx::CuContext
    chip::String
    # referenced by the host object
    availability::Vector{UInt8}
end

function ProfilerHost(f, ctx::CuContext)
    chip = chip_name(device(ctx))
    availability = counter_availability(ctx)
    params = GC.@preserve chip availability begin
        params = profiler_params(CUpti_Profiler_Host_Initialize_Params, :pHostObject;
                                 profilerType=CUPTI_PROFILER_TYPE_RANGE_PROFILER,
                                 pChipName=Base.unsafe_convert(Cstring, chip),
                                 pCounterAvailabilityImage=pointer(availability))
        cuptiProfilerHostInitialize(params)
        params
    end
    host = ProfilerHost(params[].pHostObject, ctx, chip, availability)
    try
        f(host)
    finally
        params = profiler_params(CUpti_Profiler_Host_Deinitialize_Params, :pHostObject;
                                 pHostObject=host.handle)
        cuptiProfilerHostDeinitialize(params)
    end
end

"""
    CUPTI.metric_properties(host::ProfilerHost, metric::String)

Look up the description and unit of `metric` (e.g. `dram__bytes` or `dram__bytes.sum`),
or throw a `CUPTIError` with code `CUPTI_ERROR_INVALID_METRIC_NAME` if it doesn't exist.
"""
function metric_properties(host::ProfilerHost, metric::String)
    params = GC.@preserve metric begin
        params = profiler_params(CUpti_Profiler_Host_GetMetricProperties_Params, :metricType;
                                 pHostObject=host.handle,
                                 pMetricName=Base.unsafe_convert(Cstring, metric))
        cuptiProfilerHostGetMetricProperties(params)
        params
    end
    p = params[]
    maybe_string(ptr) = ptr == C_NULL ? "" : unsafe_string(ptr)
    (; description=maybe_string(p.pDescription), hw_unit=maybe_string(p.pHwUnit),
       dim_unit=maybe_string(p.pDimUnit), type=p.metricType)
end

function add_metrics!(host::ProfilerHost, metrics::Vector{String})
    names = Base.unsafe_convert.(Cstring, metrics)
    GC.@preserve metrics names begin
        params = profiler_params(CUpti_Profiler_Host_ConfigAddMetrics_Params, :numMetrics;
                                 pHostObject=host.handle,
                                 ppMetricNames=pointer(names), numMetrics=length(names))
        unchecked_cuptiProfilerHostConfigAddMetrics(params)
    end
end

"""
    CUPTI.config_image(host::ProfilerHost, metrics::Vector{String}) -> Vector{UInt8}

Build the configuration image that makes the range profiler collect `metrics`.
"""
function config_image(host::ProfilerHost, metrics::Vector{String})
    res = add_metrics!(host, metrics)
    if res != CUPTI_SUCCESS
        # CUPTI doesn't say which metric is wrong, so try them one by one
        res in (CUPTI_ERROR_INVALID_PARAMETER, CUPTI_ERROR_INVALID_METRIC_NAME) ||
            throw_api_error(res)
        for metric in metrics
            ProfilerHost(host.ctx) do scratch
                add_metrics!(scratch, [metric]) == CUPTI_SUCCESS ||
                    throw(ArgumentError("Metric '$metric' is not available on $(host.chip)."))
            end
        end
        throw_api_error(res)
    end

    params = profiler_params(CUpti_Profiler_Host_GetConfigImageSize_Params, :configImageSize;
                             pHostObject=host.handle)
    cuptiProfilerHostGetConfigImageSize(params)
    image = Vector{UInt8}(undef, params[].configImageSize)
    params = profiler_params(CUpti_Profiler_Host_GetConfigImage_Params, :pConfigImage;
                             pHostObject=host.handle, configImageSize=length(image),
                             pConfigImage=pointer(image))
    GC.@preserve image cuptiProfilerHostGetConfigImage(params)
    return image
end

"""
    CUPTI.num_passes(config::Vector{UInt8}) -> Int

The number of times each kernel needs to be replayed to collect the metrics in `config`.
"""
function num_passes(config::Vector{UInt8})
    params = profiler_params(CUpti_Profiler_Host_GetNumOfPasses_Params, :numOfPasses;
                             configImageSize=length(config), pConfigImage=pointer(config))
    GC.@preserve config cuptiProfilerHostGetNumOfPasses(params)
    Int(params[].numOfPasses)
end

function evaluate!(values::Vector{Float64}, host::ProfilerHost,
                   counter_data::Vector{UInt8}, range::Int, metrics::Vector{String})
    names = Base.unsafe_convert.(Cstring, metrics)
    GC.@preserve counter_data metrics names values begin
        params = profiler_params(CUpti_Profiler_Host_EvaluateToGpuValues_Params, :pMetricValues;
                                 pHostObject=host.handle,
                                 pCounterDataImage=pointer(counter_data),
                                 counterDataImageSize=length(counter_data),
                                 rangeIndex=range, ppMetricNames=pointer(names),
                                 numMetrics=length(names), pMetricValues=pointer(values))
        cuptiProfilerHostEvaluateToGpuValues(params)
    end
    return values
end


## range profiler

# CUPTI supports only a single profiling session per process. this isn't a lock, as code
# being profiled that waits for another session would deadlock.
const profiler_active = Threads.Atomic{Bool}(false)

"""
    CUPTI.range_profile(f, metrics::Vector{String}; ctx=context(), max_ranges=1024)

Collect `metrics` for every kernel launched in `ctx` while executing `f()`, and return a
named tuple with a `ranges × metrics` matrix of `values` and the number of `passes` each
kernel was replayed. Ranges are numbered in the order the kernels executed, which is
serialized by CUPTI; at most `max_ranges` kernels are profiled.
"""
function range_profile(f, metrics::Vector{String}; ctx::CuContext=context(),
                       max_ranges::Integer=1024)
    Threads.atomic_cas!(profiler_active, false, true) &&
        error("Another profiling session is already active; hardware metrics cannot be collected concurrently.")
    try
        check_profiler_support(device(ctx))
        cuptiProfilerInitialize(profiler_params(CUpti_Profiler_Initialize_Params, :pPriv))
        try
            ProfilerHost(ctx) do host
                config = config_image(host, metrics)
                passes = num_passes(config)
                values = range_profile(f, host, ctx, config, metrics, max_ranges)
                (; values, passes)
            end
        finally
            cuptiProfilerDeInitialize(profiler_params(CUpti_Profiler_DeInitialize_Params, :pPriv))
        end
    finally
        profiler_active[] = false
    end
end

function range_profile(f, host::ProfilerHost, ctx::CuContext, config::Vector{UInt8},
                       metrics::Vector{String}, max_ranges::Integer)
    params = profiler_params(CUpti_RangeProfiler_Enable_Params, :pRangeProfilerObject; ctx)
    cuptiRangeProfilerEnable(params)
    profiler = params[].pRangeProfilerObject
    try
        names = Base.unsafe_convert.(Cstring, metrics)
        params = GC.@preserve metrics names begin
            params = profiler_params(CUpti_RangeProfiler_GetCounterDataSize_Params, :counterDataSize;
                                     pRangeProfilerObject=profiler,
                                     pMetricNames=pointer(names), numMetrics=length(names),
                                     maxNumOfRanges=max_ranges, maxNumRangeTreeNodes=max_ranges)
            cuptiRangeProfilerGetCounterDataSize(params)
            params
        end
        counter_data = Vector{UInt8}(undef, params[].counterDataSize)
        GC.@preserve counter_data begin
            params = profiler_params(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, :pCounterData;
                                     pRangeProfilerObject=profiler,
                                     counterDataSize=length(counter_data),
                                     pCounterData=pointer(counter_data))
            cuptiRangeProfilerCounterDataImageInitialize(params)
        end

        GC.@preserve config counter_data begin
            params = profiler_params(CUpti_RangeProfiler_SetConfig_Params, :targetNestingLevel;
                                     pRangeProfilerObject=profiler,
                                     configSize=length(config), pConfig=pointer(config),
                                     counterDataImageSize=length(counter_data),
                                     pCounterDataImage=pointer(counter_data),
                                     range=CUPTI_AutoRange, replayMode=CUPTI_KernelReplay,
                                     maxRangesPerPass=max_ranges, numNestingLevels=1,
                                     minNestingLevel=1)
            cuptiRangeProfilerSetConfig(params)

            cuptiRangeProfilerStart(profiler_params(CUpti_RangeProfiler_Start_Params, :pRangeProfilerObject;
                                                    pRangeProfilerObject=profiler))
            try
                f()
            finally
                cuptiRangeProfilerStop(profiler_params(CUpti_RangeProfiler_Stop_Params, :isAllPassSubmitted;
                                                       pRangeProfilerObject=profiler))
            end
            cuptiRangeProfilerDecodeData(profiler_params(CUpti_RangeProfiler_DecodeData_Params, :numOfRangeDropped;
                                                         pRangeProfilerObject=profiler))

            params = profiler_params(CUpti_RangeProfiler_GetCounterDataInfo_Params, :numTotalRanges;
                                     pCounterDataImage=pointer(counter_data),
                                     counterDataImageSize=length(counter_data))
            cuptiRangeProfilerGetCounterDataInfo(params)
        end
        nranges = Int(params[].numTotalRanges)

        values = Matrix{Float64}(undef, nranges, length(metrics))
        buf = Vector{Float64}(undef, length(metrics))
        for range in 1:nranges
            values[range, :] = evaluate!(buf, host, counter_data, range-1, metrics)
        end
        return values
    finally
        cuptiRangeProfilerDisable(profiler_params(CUpti_RangeProfiler_Disable_Params, :pRangeProfilerObject;
                                                  pRangeProfilerObject=profiler))
    end
end
