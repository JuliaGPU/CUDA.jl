@public has_range_profiler, check_profiler_support, ProfilerHost, metric_properties,
        range_profile, clock_status, lock_clocks

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


## clock control

has_clock_control() = isdefined(@__MODULE__, :libcupti) && version() >= v"13.4"

"""
    CUPTI.clock_status(ctx::CuContext=context()) -> Union{Nothing,CUpti_ClockControl_Status}

The clock control state of the device of `ctx`, e.g. `CUPTI_CLOCK_CONTROL_STATUS_LOCKED`,
or `nothing` if the device or CUPTI doesn't support clock control (it needs CUPTI from
CUDA 13.4 or later). On a MIG instance, this is the state of the GPU the instance belongs
to, whose clocks CUPTI can't lock but an administrator can. Wraps
`cuptiClockControlGetStatus`.
"""
function clock_status(ctx::CuContext=context())
    has_clock_control() || return nothing
    status = Ref{CUpti_ClockControl_Status}()
    res = unchecked_cuptiClockControlGetStatus(ctx, status)
    res == CUPTI_ERROR_NOT_SUPPORTED && return nothing
    res == CUPTI_SUCCESS || throw_api_error(res)
    return status[]
end

# the lock is per process, so calls that overlap (nested, or in other tasks) share the one
# the first of them took, and the last one to finish releases it
const clock_lock = ReentrantLock()
const clock_users = Ref(0)
# the device whose clocks this process locked, until unlocking them succeeds
const clock_device = Ref{Union{Nothing,CuDevice}}(nothing)
const unlock_at_exit = Ref(false)

function acquire_clocks(ctx::CuContext, mode::CUpti_ClockControl_Mode)
    @lock clock_lock begin
        dev = device(ctx)
        if clock_device[] !== nothing
            clock_device[] == dev ||
                error("Cannot lock the clocks of $dev while those of $(clock_device[]) are locked.")
            clock_users[] += 1
            return true
        end
        status = clock_status(ctx)
        if status === nothing || status in (CUPTI_CLOCK_CONTROL_STATUS_LOCKED,
                                            CUPTI_CLOCK_CONTROL_STATUS_LOCKED_TO_FLOOR)
            return false
        end
        res = unchecked_cuptiClockControlLock(ctx, mode)
        res == CUPTI_ERROR_NOT_SUPPORTED && return false
        res == CUPTI_SUCCESS || throw_api_error(res)
        clock_device[] = dev
        clock_users[] = 1
        # the lock outlives the process, so release it on `exit()` too
        if !unlock_at_exit[]
            atexit(() -> clock_device[] === nothing || unchecked_cuptiClockControlUnlock())
            unlock_at_exit[] = true
        end
        return true
    end
end

function release_clocks(; checked::Bool)
    @lock clock_lock begin
        clock_users[] -= 1
        clock_users[] == 0 || return
        res = unchecked_cuptiClockControlUnlock()
        if res == CUPTI_SUCCESS
            clock_device[] = nothing
        elseif checked
            throw_api_error(res)
        end
    end
    return
end

"""
    CUPTI.lock_clocks(f, mode=CUPTI_CLOCK_CONTROL_MODE_BASE; ctx=context())
        -> result of f()

Run `f()` with the clocks of the device of `ctx` locked to their base frequency (`mode`
`CUPTI_CLOCK_CONTROL_MODE_BASE`) or their boost frequency
(`CUPTI_CLOCK_CONTROL_MODE_BOOST`), so that GPU Boost doesn't change how fast kernels run.
The lock applies to the whole device, including other processes using it, and is released
when `f` returns or throws.

`f` runs with unlocked clocks when the device or CUPTI doesn't support clock control (see
[`CUPTI.clock_status`](@ref)), and on MIG instances, which can't lock their GPU's clocks.
Calls that overlap in this process (nested, or in other tasks) share the lock the first
one took, in its `mode`, and can't lock the clocks of another device. Clocks that CUPTI
reports as locked, e.g. by another process, are left alone: CUPTI doesn't track whose lock
that is, so unlocking would release it. This check races with other processes locking the
clocks at the same time. Clocks locked with `nvidia-smi --lock-gpu-clocks` are not reported
as locked; the lock taken here overrides them while `f` runs.

If the process crashes while `f` runs, the clocks stay locked, which `nvidia-smi
--reset-gpu-clocks` doesn't undo. Locking and unlocking them through CUPTI does:

```julia
CUPTI.cuptiClockControlLock(context(), CUPTI.CUPTI_CLOCK_CONTROL_MODE_BASE)
CUPTI.cuptiClockControlUnlock()
```

Wraps `cuptiClockControlLock`.
"""
function lock_clocks(f, mode::CUpti_ClockControl_Mode=CUPTI_CLOCK_CONTROL_MODE_BASE;
                     ctx::CuContext=context())
    # an interrupt between taking the lock and running `f` would leave the clocks locked
    owned = Ref(false)
    failed = true
    try
        disable_sigint(() -> owned[] = acquire_clocks(ctx, mode))
        result = f()
        failed = false
        return result
    finally
        # failing to unlock shouldn't hide why `f` failed
        checked = !failed
        owned[] && disable_sigint(() -> release_clocks(; checked))
    end
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
    CUPTI.check_profiler_support(dev::CuDevice; api=CUPTI_PROFILER_RANGE_PROFILING,
                                 feature="hardware metrics") -> nothing

Throw an error explaining why `dev` doesn't support the profiling `api`, if it doesn't.
`feature` names what the API is used for in the error message. Wraps
`cuptiProfilerDeviceSupported`.
"""
function check_profiler_support(dev::CuDevice; api::CUpti_Profiler_API=CUPTI_PROFILER_RANGE_PROFILING,
                                feature::String="hardware metrics")
    params = profiler_params(CUpti_Profiler_DeviceSupported_Params, :api;
                             cuDevice=CUdevice(deviceid(dev)), api)
    res = unchecked_cuptiProfilerDeviceSupported(params)
    # MIG devices fail the query itself
    res == CUPTI_ERROR_INVALID_PARAMETER &&
        error("This device does not support $feature (MIG instances are not supported).")
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
    error("This device does not support $feature: $(join(reasons, ", ")).")
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
    CUPTI.ProfilerHost(f, ctx::CuContext) -> result of f(host)

Create a profiler host object for the device of `ctx`, which configures and evaluates
range profiler metrics, pass it to `f`, and return the result of `f`. The object is only
valid while `f` runs, and its fields are not part of the API. Wraps
`cuptiProfilerHostInitialize`, which needs CUPTI from CUDA 12.6 or later.
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
        -> @NamedTuple{description::String, hw_unit::String, dim_unit::String,
                       type::CUpti_MetricType}

Look up the description and unit of `metric` (e.g. `dram__bytes` or `dram__bytes.sum`),
or throw a `CUPTIError` with code `CUPTI_ERROR_INVALID_METRIC_NAME` if it doesn't exist.
Wraps `cuptiProfilerHostGetMetricProperties`, which needs CUPTI from CUDA 12.6 or
later.
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

# CUPTI 13.2 and later lose ranges when only timestamp metrics (e.g. gpu__time_duration) are
# collected, and can break the range profiler for the rest of the process. collecting a
# single-pass hardware counter as well avoids that (NVIDIA bug, JuliaGPU/CUDA.jl#3595).
with_hardware_counter(metrics) =
    version() >= v"13.2" && !isempty(metrics) && all(startswith("gpu__time_"), metrics) ?
        [metrics; "smsp__warps_launched.sum"] : metrics

"""
    CUPTI.range_profile(f, metrics::Vector{String}; ctx=context(), max_ranges=1024,
                        clock_control=nothing, cache_control=CUPTI_CACHE_CONTROL_MODE_NONE)
        -> @NamedTuple{values::Matrix{Float64}, passes::Int}

Collect `metrics` for every kernel launched in `ctx` while executing `f()`, and return a
named tuple with a `ranges × metrics` matrix of `values` and the number of `passes` each
kernel was replayed. Ranges are numbered in the order the kernels executed, which is
serialized by CUPTI; at most `max_ranges` kernels are profiled.

To make results repeatable, `clock_control` can lock the clocks during the session to
`CUPTI_CLOCK_CONTROL_MODE_BASE` or `CUPTI_CLOCK_CONTROL_MODE_BOOST` (see
[`CUPTI.lock_clocks`](@ref)), and `cache_control` can reset caches before every replay
pass: all of them (`CUPTI_CACHE_CONTROL_MODE_ALL`) or only L2
(`CUPTI_CACHE_CONTROL_MODE_L2`). Both need CUPTI from CUDA 13.4 or later, and are ignored
by older versions.

This uses CUPTI's range profiler in auto-range, kernel-replay mode, which needs CUPTI from
CUDA 12.6 Update 2 or later (see [`CUPTI.has_range_profiler`](@ref)) and a device that
supports it (see [`CUPTI.check_profiler_support`](@ref)). Only one profiling session,
of this function or [`CUPTI.pc_sample`](@ref), can be active at a time.
"""
function range_profile(f, metrics::Vector{String}; ctx::CuContext=context(),
                       max_ranges::Integer=1024,
                       clock_control::Union{Nothing,CUpti_ClockControl_Mode}=nothing,
                       cache_control::CUpti_CacheControl_Mode=CUPTI_CACHE_CONTROL_MODE_NONE)
    Threads.atomic_cas!(profiler_active, false, true) &&
        error("Another profiling session is already active; hardware metrics cannot be collected concurrently.")
    try
        check_profiler_support(device(ctx))
        function session()
            cuptiProfilerInitialize(profiler_params(CUpti_Profiler_Initialize_Params, :pPriv))
            try
                ProfilerHost(ctx) do host
                    collected = with_hardware_counter(metrics)
                    config = config_image(host, collected)
                    passes = num_passes(config)
                    values = range_profile(f, host, ctx, config, collected, max_ranges,
                                           cache_control)
                    (; values=values[:, 1:length(metrics)], passes)
                end
            finally
                cuptiProfilerDeInitialize(profiler_params(CUpti_Profiler_DeInitialize_Params, :pPriv))
            end
        end
        # CUPTI expects clocks to be locked before the session is enabled
        clock_control === nothing ? session() : lock_clocks(session, clock_control; ctx)
    finally
        profiler_active[] = false
    end
end

function range_profile(f, host::ProfilerHost, ctx::CuContext, config::Vector{UInt8},
                       metrics::Vector{String}, max_ranges::Integer,
                       cache_control::CUpti_CacheControl_Mode)
    # older CUPTI rejects the struct size that includes the cache control mode
    params = if version() >= v"13.4"
        profiler_params(CUpti_RangeProfiler_Enable_Params, :cacheControlMode; ctx,
                        cacheControlMode=cache_control)
    else
        profiler_params(CUpti_RangeProfiler_Enable_Params, :pRangeProfilerObject; ctx)
    end
    cuptiRangeProfilerEnable(params)
    profiler = params[].pRangeProfilerObject
    values = try
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
        values
    catch
        # after a failed session, disabling can fail too, which shouldn't hide the actual error
        unchecked_cuptiRangeProfilerDisable(profiler_params(CUpti_RangeProfiler_Disable_Params, :pRangeProfilerObject;
                                                            pRangeProfilerObject=profiler))
        rethrow()
    end
    cuptiRangeProfilerDisable(profiler_params(CUpti_RangeProfiler_Disable_Params, :pRangeProfilerObject;
                                              pRangeProfilerObject=profiler))
    return values
end
