# https://docs.nvidia.com/cupti/api/group__CUPTI__VERSION__API.html
const cupti_versions = [
    v"4.0",
    v"4.1",
    v"5.0",
    v"5.5",
    v"6.0",
    v"6.5",
    v"6.5.1", # with sm_52 support
    v"7.0",
    v"8.0",
    v"9.0",
    v"9.1",
    v"10.0", # and v10.1 and v10.2
    v"11.0",
    v"11.1",
    v"11.2", # and v11.3 and v11.4
    v"11.5",
    v"11.6",
    v"11.8",
    v"12.0",
    v"12.2",
    v"12.3",
    v"12.4",
    v"12.5",
    v"12.6",
    v"12.7",
    v"12.8",
    v"12.9",
    v"12.9.1"]

function version()
    version_ref = Ref{Cuint}()
    cuptiGetVersion(version_ref)
    if CUDA.runtime_version() < v"13"
        cupti_versions[version_ref[]]
    else
        major, ver = divrem(version_ref[], 10000)
        minor, patch = divrem(ver, 100)
        VersionNumber(major, minor, patch)
    end
end

# XXX: `cuptiGetVersion` returns something more like the API version, and doesn't change
#      in between update releases (even when those contain functional changes/fixes)...
#      NVIDIA suggests to instead use the version number as attached to the filename.
function library_version()
    filename = basename(realpath(libcupti))
    rx = if Sys.islinux()
        r"libcupti.so.(\d+)\.(\d+)\.(\d+)"
    elseif Sys.iswindows()
        r"cupti64_(\d+).(\d+).(\d+).dll"
    else
        error("Unsupported platform; please file an issue with the following information:\n$(sprint(versioninfo))")
    end
    m = match(rx, filename)
    if m === nothing
        error("Could not extract version number from CUPTI library `$filename`; please file an issue.")
    end
    VersionNumber(parse(Int, m.captures[1]),
                  parse(Int, m.captures[2]),
                  parse(Int, m.captures[3]))
end


#
# callback API
#

# multiple subscribers aren't supported, so make sure we only call CUPTI once
const callback_lock = ReentrantLock()

function callback(userdata::Ptr{Cvoid}, domain::CUpti_CallbackDomain,
                  id::CUpti_CallbackId, data_ptr::Ptr{Cvoid})
    cfg = Base.unsafe_pointer_to_objref(userdata)::CallbackConfig

    # decode the callback data
    datatype = if domain in (CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_CB_DOMAIN_RUNTIME_API)
        CUpti_CallbackData
    elseif domain == CUPTI_CB_DOMAIN_RESOURCE
        CUpti_ResourceData
    elseif domain == CUPTI_CB_DOMAIN_SYNCHRONIZE
        CUpti_SynchronizeData
    elseif domain == CUPTI_CB_DOMAIN_NVTX
        CUpti_NvtxData
    else
        @warn """Unsupported callback domain: $(domain).
                 Please file an issue, or extend the implementation of `CUPTI.callback` to handle this callback kind."""
        return
    end
    data = unsafe_load(convert(Ptr{datatype}, data_ptr))

    # invoke the actual user callback
    cfg.callback(domain, id, data)

    return
end

"""
    cfg = CUPTI.CallbackConfig(callback_kinds) do domain, id, data
        # inspect data
    end

    CUPTI.enable!(cfg) do
        # do stuff
    end
"""
mutable struct CallbackConfig
    callback::Function
    callback_kinds::Vector{CUpti_CallbackDomain}
end

function enable!(f::Base.Callable, cfg::CallbackConfig)
    @lock callback_lock begin
        callback_ptr =
            @cfunction(callback, Cvoid,
                       (Ptr{Cvoid}, CUpti_CallbackDomain, CUpti_CallbackId, Ptr{Cvoid}))

        GC.@preserve cfg begin
            # set-up subscriber
            subscriber_ref = Ref{CUpti_SubscriberHandle}()
            cuptiSubscribe(subscriber_ref, callback_ptr, Base.pointer_from_objref(cfg))
            subscriber = subscriber_ref[]

            # enable domains
            for callback_kind in cfg.callback_kinds
                CUPTI.cuptiEnableDomain(true, subscriber, callback_kind)
            end

            try
                f()
            finally
                # disable callback kinds
                for callback_kind in cfg.callback_kinds
                    CUPTI.cuptiEnableDomain(false, subscriber, callback_kind)
                end

                # disable the subscriber
                CUPTI.cuptiUnsubscribe(subscriber)
            end
        end
    end
end


#
# activity API
#

"""
    cfg = CUPTI.ActivityConfig(activity_kinds)

    CUPTI.enable!(cfg) do
        # do stuff
    end

    CUPTI.process(cfg) do ctx, stream_id, record
        # inspect record
    end

High-level interface to the CUPTI activity API.
"""
struct ActivityConfig
    activity_kinds::Vector{CUpti_ActivityKind}

    available_buffers::Vector{Vector{UInt8}}
    active_buffers::Vector{Vector{UInt8}}

    results::Vector{Any}
end

function ActivityConfig(activity_kinds)
    # pre-allocate a couple of buffers to avoid allocating while profiling
    available_buffers = [allocate_buffer() for _ in 1:5]

    ActivityConfig(activity_kinds, available_buffers, Vector{UInt8}[], [])
end

function allocate_buffer()
    # "for typical workloads, it's suggested to choose a size between 1 and 10 MB."
    Array{UInt8}(undef, 8 * 1024 * 1024)  # 8 MB
end

const activity_lock = ReentrantLock()
const activity_config = Ref{Union{Nothing,ActivityConfig}}(nothing)

function request_buffer(dest_ptr, sz_ptr, max_num_records_ptr)
    # this function is called by CUPTI, but directly from the application, so it should be
    # fine to perform I/O or allocate memory here.

    dest = Base.unsafe_wrap(Array, dest_ptr, 1)
    sz = Base.unsafe_wrap(Array, sz_ptr, 1)
    max_num_records = Base.unsafe_wrap(Array, max_num_records_ptr, 1)

    cfg = activity_config[]
    if cfg !== nothing
        buf = if isempty(cfg.available_buffers)
            allocate_buffer()
        else
            pop!(cfg.available_buffers)
        end
        ptr = pointer(buf)
        push!(cfg.active_buffers, buf)
        sizehint!(cfg.results, length(cfg.active_buffers))

        dest[] = pointer(buf)
        sz[] = sizeof(buf)
        max_num_records[] = 0
    else
        dest[] = C_NULL
        sz[] = 0
        max_num_records[] = 0
    end

    return
end

function complete_buffer(ctx_handle, stream_id, buf_ptr, sz, valid_sz)
    # this function is called by a CUPTI worker thread while our application may be waiting
    # for `cuptiActivityFlushAll` to complete. that means we cannot do I/O here, or we could
    # yield while the application cannot make any progress.
    #
    # we also may not trigger GC, because the main application cannot reach a safepoint.
    # to prevent this, we call `sizehint!` in `request_buffer`.
    # XXX: `sizehint!` isn't a guarantee; use `resize!` and a cursor?

    cfg = activity_config[]
    if cfg !== nothing
        push!(cfg.results, (ctx_handle, stream_id, buf_ptr, sz, valid_sz))
    end

    return
end

function enable!(f::Base.Callable, cfg::ActivityConfig)
    @lock activity_lock begin
        activity_config[] = cfg

        # set-up callbacks
        request_buffer_ptr =
            @cfunction(request_buffer, Cvoid,
                       (Ptr{Ptr{UInt8}}, Ptr{Csize_t}, Ptr{Csize_t}))
        complete_buffer_ptr =
            @cfunction(complete_buffer, Cvoid,
                       (CUDA.CUcontext, UInt32, Ptr{UInt8}, Csize_t, Csize_t))
        cuptiActivityRegisterCallbacks(request_buffer_ptr, complete_buffer_ptr)

        activity_config[] = cfg

        # enable requested activity kinds
        for activity_kind in cfg.activity_kinds
            cuptiActivityEnable(activity_kind)
        end

        try
            f()
        finally
            # disable activity kinds
            for activity_kind in cfg.activity_kinds
                cuptiActivityDisable(activity_kind)
            end

            # flush all activity records, even incomplete ones
            cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)

            activity_config[] = nothing
        end
    end
end

function process(f, cfg::ActivityConfig)
    activity_types = Dict(
        CUPTI_ACTIVITY_KIND_DRIVER              => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_RUNTIME             => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_NAME                => CUpti_ActivityName,
        CUPTI_ACTIVITY_KIND_MARKER              => CUpti_ActivityMarker2,
        CUPTI_ACTIVITY_KIND_MARKER_DATA         => CUpti_ActivityMarkerData,
    )
    # NOTE: the CUPTI version is unreliable, e.g., both CUDA 11.5 and 11.6 have CUPTI 16,
    #       yet CUpti_ActivityMemset4 is only available in CUDA 11.6.
    cuda_version = CUDA.runtime_version()
    ## kernel activities
    activity_types[CUPTI_ACTIVITY_KIND_KERNEL] =
        CUpti_ActivityKernel9
    activity_types[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL] =
        activity_types[CUPTI_ACTIVITY_KIND_KERNEL]
    ## memcpy activities
    activity_types[CUPTI_ACTIVITY_KIND_MEMCPY] =
        CUpti_ActivityMemcpy5
    activity_types[CUPTI_ACTIVITY_KIND_MEMSET] =
        CUpti_ActivityMemset4
    activity_types[CUPTI_ACTIVITY_KIND_MEMORY2] =
        CUpti_ActivityMemory3

    # extract typed activity records
    for (ctx_handle, stream_id, buf_ptr, sz, valid_sz) in cfg.results
        ctx = ctx_handle == C_NULL ? nothing : CuContext(ctx_handle)
        # XXX: can we reconstruct the stream from the stream ID?

        record_ptr = Ref{Ptr{CUpti_Activity}}(C_NULL)
        while true
            try
                cuptiActivityGetNextRecord(buf_ptr, valid_sz, record_ptr)
                record = unsafe_load(record_ptr[])

                if haskey(activity_types, record.kind)
                    typ = activity_types[record.kind]
                    typed_ptr = convert(Ptr{typ}, record_ptr[])
                    typed_record = unsafe_load(typed_ptr)
                    f(ctx, stream_id, typed_record)
                else
                    @warn """Unsupported activity kind: $(record.kind).
                             Please file an issue, or extend the implementation of `CUPTI.process` to handle this activity kind."""
                end
            catch err
                if isa(err, CUPTIError) && err.code == CUPTI_ERROR_MAX_LIMIT_REACHED
                    break
                end
                rethrow()
            end
        end
    end
end

#
# profiler host API
#

"""
    _check_profiler_host_api()

Check that the CUPTI Profiler Host API is available (requires CUDA >= 12.6).
"""
function _check_profiler_host_api()
    if CUDA.runtime_version() < v"12.6"
        error("CUPTI Profiler Host API requires CUDA >= 12.6 (got $(CUDA.runtime_version()))")
    end
end

"""
    profiler_device_supported(; dev=CUDA.device(), api=CUPTI_PROFILER_RANGE_PROFILING) -> Bool

Check whether the CUPTI profiler APIs are supported on the given device.
Returns `false` for MIG partitions, unsupported architectures, vGPU, etc.
"""
function profiler_device_supported(;
        dev::CUDA.CuDevice=CUDA.device(),
        api::CUpti_Profiler_API=CUPTI_PROFILER_RANGE_PROFILING)
    _check_profiler_host_api()
    try
        params = Ref(CUpti_Profiler_DeviceSupported_Params(
            @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_DeviceSupported_Params, sku),
            C_NULL,
            dev,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
            api,
            CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        ))
        cuptiProfilerDeviceSupported(params)
        return params[].isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED
    catch e
        # CUPTI_ERROR_INVALID_PARAMETER on MIG devices — profiling not supported
        if e isa CUPTIError
            return false
        end
        rethrow()
    end
end

# compute capability → chip name mapping
# from cuptiProfilerHostGetSupportedChips() output
const CC_TO_CHIP = Dict{VersionNumber,String}(
    v"7.5" => "TU102",
    v"8.0" => "GA100",
    v"8.6" => "GA102",
    v"8.9" => "AD102",
    v"9.0" => "GH100",
    v"10.0" => "GB100",
    v"10.2" => "GB202",
    v"11.0" => "GB110",
)

"""
    check_profiling_permissions()

Check if CUPTI profiling permissions are available on Linux.
PM sampling and hardware counter collection require
`NVreg_RestrictProfilingToAdminUsers=0`.
"""
function check_profiling_permissions()
    if !Sys.islinux()
        return true
    end
    nvidia_params = "/proc/driver/nvidia/params"
    if isfile(nvidia_params)
        content = read(nvidia_params, String)
        if contains(content, "RmProfilingAdminOnly: 1")
            @warn """CUPTI hardware counter collection requires profiling permissions.
                     Set NVreg_RestrictProfilingToAdminUsers=0 in /etc/modprobe.d/nvidia-profiler.conf
                     and reload the nvidia kernel module, or run as root."""
            return false
        end
    end
    return true
end

"""
    chip_name(dev::CUDA.CuDevice) -> String

Get the CUPTI chip name for a CUDA device.
Falls back to querying supported chips if the compute capability
is not in the built-in mapping.
"""
function chip_name(dev::CUDA.CuDevice)
    cc = CUDA.capability(dev)
    if haskey(CC_TO_CHIP, cc)
        return CC_TO_CHIP[cc]
    end
    # fallback: query supported chips and match by CC prefix
    chips = supported_chips()
    # try exact match first, then look for any chip
    for chip in chips
        return chip  # return first available as fallback
    end
    error("Could not determine chip name for compute capability $cc. " *
          "Supported chips: $(join(chips, ", "))")
end

"""
    supported_chips() -> Vector{String}

List all GPU chip names supported by the CUPTI profiler host API.
"""
function supported_chips()
    _check_profiler_host_api()
    params = Ref(CUpti_Profiler_Host_GetSupportedChips_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSupportedChips_Params, ppChipNames),
        C_NULL, 0, Ptr{Cstring}(0),
    ))
    cuptiProfilerHostGetSupportedChips(params)
    p = params[]
    return [unsafe_string(unsafe_load(p.ppChipNames, i)) for i in 1:p.numChips]
end

"""
    ProfilerHostContext

Manages a CUPTI profiler host object for metric enumeration and
config image creation. Supports both range profiling and PM sampling.

```julia
ctx = CUPTI.ProfilerHostContext("GH100"; profiler_type=CUPTI_PROFILER_TYPE_PM_SAMPLING)
metrics = CUPTI.base_metrics(ctx, CUPTI_METRIC_TYPE_COUNTER)
close(ctx)
```
"""
mutable struct ProfilerHostContext
    host_object::Ptr{CUpti_Profiler_Host_Object}
    chip_name::String
    profiler_type::CUpti_ProfilerType

    function ProfilerHostContext(chip::String;
                                 profiler_type::CUpti_ProfilerType=CUPTI_PROFILER_TYPE_PM_SAMPLING,
                                 counter_availability_image::Union{Nothing,Vector{UInt8}}=nothing,
                                 single_pass_set_name::Union{Nothing,String}=nothing)
        params = Ref(CUpti_Profiler_Host_Initialize_Params(
            @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Initialize_Params, pSinglePassMetricSetName),
            C_NULL,
            profiler_type,
            Base.unsafe_convert(Cstring, chip),
            counter_availability_image === nothing ? Ptr{UInt8}(0) : pointer(counter_availability_image),
            Ptr{CUpti_Profiler_Host_Object}(0),  # pHostObject (out)
            single_pass_set_name === nothing ? Cstring(C_NULL) : Base.unsafe_convert(Cstring, single_pass_set_name),
        ))
        cuptiProfilerHostInitialize(params)
        obj = new(params[].pHostObject, chip, profiler_type)
        finalizer(obj) do o
            if o.host_object != C_NULL
                deinit = Ref(CUpti_Profiler_Host_Deinitialize_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Deinitialize_Params, pHostObject),
                    C_NULL,
                    o.host_object,
                ))
                cuptiProfilerHostDeinitialize(deinit)
                o.host_object = C_NULL
            end
        end
        return obj
    end
end

function Base.close(ctx::ProfilerHostContext)
    finalize(ctx)
end

"""
    base_metrics(ctx::ProfilerHostContext, metric_type::CUpti_MetricType) -> Vector{String}

List all base metrics of the given type (CUPTI_METRIC_TYPE_COUNTER,
CUPTI_METRIC_TYPE_RATIO, or CUPTI_METRIC_TYPE_THROUGHPUT).
"""
function base_metrics(ctx::ProfilerHostContext, metric_type::CUpti_MetricType)
    params = Ref(CUpti_Profiler_Host_GetBaseMetrics_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetBaseMetrics_Params, numMetrics),
        C_NULL,
        ctx.host_object,
        metric_type,
        Ptr{Cstring}(0), 0,
    ))
    cuptiProfilerHostGetBaseMetrics(params)
    p = params[]
    return [unsafe_string(unsafe_load(p.ppMetricNames, i)) for i in 1:p.numMetrics]
end

"""
    sub_metrics(ctx::ProfilerHostContext, metric_name::String, metric_type::CUpti_MetricType) -> Vector{String}

List available sub-metrics (rollups like `.sum`, `.avg`, `.pct`, etc.)
for a given base metric.
"""
function sub_metrics(ctx::ProfilerHostContext, metric_name::String,
                     metric_type::CUpti_MetricType)
    params = Ref(CUpti_Profiler_Host_GetSubMetrics_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSubMetrics_Params, ppSubMetrics),
        C_NULL,
        ctx.host_object,
        metric_type,
        Base.unsafe_convert(Cstring, metric_name),
        0, Ptr{Cstring}(0),
    ))
    cuptiProfilerHostGetSubMetrics(params)
    p = params[]
    return [unsafe_string(unsafe_load(p.ppSubMetrics, i)) for i in 1:p.numOfSubmetrics]
end

"""
    MetricProperties

Properties of a CUPTI metric.
"""
struct MetricProperties
    description::String
    hw_unit::String
    dim_unit::String
    metric_type::CUpti_MetricType
    collection_scope::CUpti_MetricCollectionScope
end

"""
    metric_properties(ctx::ProfilerHostContext, metric_name::String) -> MetricProperties

Get the description, hardware unit, and other properties of a metric.
"""
function metric_properties(ctx::ProfilerHostContext, metric_name::String)
    params = Ref(CUpti_Profiler_Host_GetMetricProperties_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetMetricProperties_Params, metricCollectionScope),
        C_NULL,
        ctx.host_object,
        Base.unsafe_convert(Cstring, metric_name),
        C_NULL, C_NULL, C_NULL,
        CUPTI_METRIC_TYPE_COUNTER,
        CUPTI_METRIC_COLLECTION_SCOPE_CONTEXT,
    ))
    cuptiProfilerHostGetMetricProperties(params)
    p = params[]
    return MetricProperties(
        p.pDescription == C_NULL ? "" : unsafe_string(p.pDescription),
        p.pHwUnit == C_NULL ? "" : unsafe_string(p.pHwUnit),
        p.pDimUnit == C_NULL ? "" : unsafe_string(p.pDimUnit),
        p.metricType,
        p.metricCollectionScope,
    )
end

"""
    single_pass_sets(chip::String) -> Vector{String}

List the single-pass metric set names available for a chip
(e.g. "TriageCompute" on Hopper).

Requires CUDA >= 13.1. Returns an empty vector on older versions.
"""
function single_pass_sets(chip::String)
    _check_profiler_host_api()
    # cuptiProfilerHostGetSinglePassSets was added in CUDA 13.2
    if CUDA.runtime_version() < v"13.2"
        return String[]
    end

    # first call: query count
    params = Ref(CUpti_Profiler_Host_GetSinglePassSets_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSinglePassSets_Params, ppSinglePassSets),
        C_NULL,
        Base.unsafe_convert(Cstring, chip),
        0, C_NULL,
    ))
    cuptiProfilerHostGetSinglePassSets(params)
    n = params[].numOfSinglePassSets
    if n == 0
        return String[]
    end
    # second call: get names
    buf = Vector{Cstring}(undef, n)
    params = Ref(CUpti_Profiler_Host_GetSinglePassSets_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSinglePassSets_Params, ppSinglePassSets),
        C_NULL,
        Base.unsafe_convert(Cstring, chip),
        n, pointer(buf),
    ))
    cuptiProfilerHostGetSinglePassSets(params)
    return [unsafe_string(buf[i]) for i in 1:n]
end

"""
    list_metrics(; chip=nothing, type=nothing) -> Vector{@NamedTuple{name::String, description::String, hw_unit::String}}

List available PM sampling metrics for a GPU. If `chip` is not specified,
auto-detects from the current CUDA device.

`type` can be `CUPTI_METRIC_TYPE_COUNTER`, `CUPTI_METRIC_TYPE_RATIO`,
or `CUPTI_METRIC_TYPE_THROUGHPUT`. If `nothing`, lists all types.
"""
function list_metrics(; chip::Union{String,Nothing}=nothing,
                       type::Union{CUpti_MetricType,Nothing}=nothing)
    if chip === nothing
        chip = chip_name(CUDA.device())
    end
    ctx = ProfilerHostContext(chip; profiler_type=CUPTI_PROFILER_TYPE_PM_SAMPLING)
    try
        types = type === nothing ?
            [CUPTI_METRIC_TYPE_COUNTER, CUPTI_METRIC_TYPE_RATIO, CUPTI_METRIC_TYPE_THROUGHPUT] :
            [type]
        results = @NamedTuple{name::String, description::String, hw_unit::String}[]
        for t in types
            for name in base_metrics(ctx, t)
                props = metric_properties(ctx, name)
                push!(results, (name=name, description=props.description, hw_unit=props.hw_unit))
            end
        end
        return results
    finally
        close(ctx)
    end
end

"""
    metric_info(metric_name::String; chip=nothing)

Print detailed information about a metric including its sub-metrics.
"""
function metric_info(metric_name::String; chip::Union{String,Nothing}=nothing)
    if chip === nothing
        chip = chip_name(CUDA.device())
    end
    ctx = ProfilerHostContext(chip; profiler_type=CUPTI_PROFILER_TYPE_PM_SAMPLING)
    try
        props = metric_properties(ctx, metric_name)
        println("Metric: $metric_name")
        println("  Description: $(props.description)")
        println("  HW Unit:     $(props.hw_unit)")
        println("  Dim Unit:    $(props.dim_unit)")
        println("  Type:        $(props.metric_type)")
        println("  Scope:       $(props.collection_scope)")
        subs = sub_metrics(ctx, metric_name, props.metric_type)
        if !isempty(subs)
            println("  Sub-metrics ($(length(subs))):")
            for s in subs
                println("    .$s")
            end
        end
    finally
        close(ctx)
    end
end


#
# config image helpers (shared by range profiler and PM sampling)
#

const profiler_lock = ReentrantLock()

function _profiler_initialize()
    params = Ref(CUpti_Profiler_Initialize_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Initialize_Params, pPriv),
        C_NULL,
    ))
    cuptiProfilerInitialize(params)
end

"""
    _get_counter_availability(device_index) -> Vector{UInt8}

Query the counter availability image for a device. Required by PM sampling
for forward chip compatibility.
"""
function _get_counter_availability(device_index::Int=0)
    # Query size
    params = Ref(CUpti_PmSampling_GetCounterAvailability_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterAvailability_Params, pCounterAvailabilityImage),
        C_NULL, device_index, 0, Ptr{UInt8}(0),
    ))
    cuptiPmSamplingGetCounterAvailability(params)
    avail_size = params[].counterAvailabilityImageSize

    # Retrieve image
    avail_image = Vector{UInt8}(undef, avail_size)
    params = Ref(CUpti_PmSampling_GetCounterAvailability_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterAvailability_Params, pCounterAvailabilityImage),
        C_NULL, device_index, avail_size, pointer(avail_image),
    ))
    cuptiPmSamplingGetCounterAvailability(params)
    return avail_image
end

"""
    _with_profiler_host(f, metric_names, profiler_type; chip, counter_availability_image)

Shared setup for range profiling and PM sampling: initialize CUPTI,
create a ProfilerHostContext, configure metrics, build config image,
then call `f(host_ctx, config_image)`.
"""
function _with_profiler_host(f, metric_names::Vector{String},
                             profiler_type::CUpti_ProfilerType;
                             chip::Union{String,Nothing}=nothing,
                             counter_availability_image::Union{Nothing,Vector{UInt8}}=nothing)
    _check_profiler_host_api()
    check_profiling_permissions()

    if chip === nothing
        chip = chip_name(CUDA.device())
    end

    @lock profiler_lock begin
        _profiler_initialize()

        host_ctx = ProfilerHostContext(chip;
            profiler_type,
            counter_availability_image)
        try
            config_add_metrics!(host_ctx, metric_names)
            config_image = get_config_image(host_ctx)
            return f(host_ctx, config_image)
        finally
            close(host_ctx)
        end
    end
end

"""
    config_add_metrics!(ctx::ProfilerHostContext, metric_names::Vector{String})

Add metrics to the profiler host config. Must be called before
`get_config_image`.
"""
function config_add_metrics!(ctx::ProfilerHostContext, metric_names::Vector{String})
    c_names = Base.unsafe_convert.(Cstring, metric_names)
    GC.@preserve metric_names c_names begin
        params = Ref(CUpti_Profiler_Host_ConfigAddMetrics_Params(
            @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_ConfigAddMetrics_Params, numMetrics),
            C_NULL,
            ctx.host_object,
            pointer(c_names),
            length(c_names),
        ))
        cuptiProfilerHostConfigAddMetrics(params)
    end
end

"""
    get_config_image(ctx::ProfilerHostContext) -> Vector{UInt8}

Get the serialized config image for the currently configured metrics.
"""
function get_config_image(ctx::ProfilerHostContext)
    # get size
    size_params = Ref(CUpti_Profiler_Host_GetConfigImageSize_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetConfigImageSize_Params, configImageSize),
        C_NULL,
        ctx.host_object,
        0,
    ))
    cuptiProfilerHostGetConfigImageSize(size_params)
    img_size = size_params[].configImageSize

    # get image
    config_image = Vector{UInt8}(undef, img_size)
    img_params = Ref(CUpti_Profiler_Host_GetConfigImage_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetConfigImage_Params, pConfigImage),
        C_NULL,
        ctx.host_object,
        img_size,
        pointer(config_image),
    ))
    cuptiProfilerHostGetConfigImage(img_params)
    return config_image
end

"""
    get_num_passes(config_image::Vector{UInt8}) -> Int

Return the number of profiling passes required for the given config.
"""
function get_num_passes(config_image::Vector{UInt8})
    params = Ref(CUpti_Profiler_Host_GetNumOfPasses_Params(
        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetNumOfPasses_Params, numOfPasses),
        C_NULL,
        length(config_image),
        pointer(config_image),
        0,
    ))
    cuptiProfilerHostGetNumOfPasses(params)
    return Int(params[].numOfPasses)
end

"""
    evaluate_metrics(ctx::ProfilerHostContext, counter_data::Vector{UInt8},
                     range_index::Int, metric_names::Vector{String}) -> Vector{Float64}

Evaluate hardware counter data for a given range/sample index.
"""
function evaluate_metrics(ctx::ProfilerHostContext, counter_data::Vector{UInt8},
                          range_index::Int, metric_names::Vector{String})
    values = Vector{Float64}(undef, length(metric_names))
    c_names = Base.unsafe_convert.(Cstring, metric_names)
    GC.@preserve metric_names c_names counter_data values begin
        params = Ref(CUpti_Profiler_Host_EvaluateToGpuValues_Params(
            @CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_EvaluateToGpuValues_Params, pMetricValues),
            C_NULL,
            ctx.host_object,
            pointer(counter_data),
            length(counter_data),
            range_index,
            pointer(c_names),
            length(c_names),
            pointer(values),
        ))
        cuptiProfilerHostEvaluateToGpuValues(params)
    end
    return values
end


#
# range profiler API
#

"""
    RangeProfileResult

Result from range profiling a kernel or code region.
"""
struct RangeProfileResult
    range_names::Vector{String}
    kernel_names::Vector{String}  # from callback API (may be empty)
    metric_names::Vector{String}
    values::Matrix{Float64}  # ranges × metrics
end

"""
    range_profile(f, metric_names::Vector{String};
                  range_mode=CUPTI_AutoRange,
                  max_ranges=64,
                  max_nesting=1) -> RangeProfileResult

Profile hardware counters for GPU kernels launched within `f()`.

With `CUPTI_AutoRange`, each kernel launch becomes a separate range.
With `CUPTI_UserRange`, use `push_range!`/`pop_range!` to define custom ranges.

Uses CUPTI's KernelReplay mode: when metrics require multiple passes,
CUPTI internally replays each kernel, so `f()` is only called once.

!!! warning
    CUPTI's KernelReplay mode internally re-executes GPU kernels to
    collect multi-pass metrics. While `f()` itself runs only once,
    individual kernels may be replayed by CUPTI. Ensure kernels
    produce deterministic results.

```julia
result = CUPTI.range_profile(["sm__cycles_active.avg", "dram__throughput.avg.pct_of_peak_sustained_elapsed"]) do
    CUDA.@sync my_kernel(args...)
end
```
"""
function range_profile(f, metric_names::Vector{String};
                       chip::Union{String,Nothing}=nothing,
                       range_mode::CUpti_ProfilerRange=CUPTI_AutoRange,
                       max_ranges::Int=64,
                       max_nesting::Int=1)
    # Range profiling flow:
    #
    # 1. _with_profiler_host creates a ProfilerHostContext, configures metrics,
    #    and builds a config image (shared with PM sampling).
    #
    # 2. Enable the range profiler on the CUDA context — this creates a device-side
    #    profiler object that intercepts kernel launches.
    #
    # 3. Allocate a counter data image — buffer where CUPTI writes raw counter values.
    #
    # 4. SetConfig + Start + run user code + Stop + DecodeData:
    #    KernelReplay mode: CUPTI internally re-executes each kernel as many times
    #    as needed for multi-pass collection. f() is only called once.
    #
    # 5. Evaluate: host context converts raw counter data into metric values.
    #
    # Kernel name capture:
    #    Auto-range mode names ranges "0", "1"... We use the CUPTI callback API
    #    (fires synchronously on host thread) to capture symbolName from each
    #    kernel launch. Callbacks coexist with the range profiler.

    # Set up callback to capture kernel names from driver API calls
    kernel_names = String[]
    cb_cfg = CallbackConfig([CUPTI_CB_DOMAIN_DRIVER_API]) do domain, id, data
        if data.callbackSite == CUPTI_API_ENTER && data.symbolName != C_NULL
            name = unsafe_string(data.symbolName)
            if name != "Unknown"
                push!(kernel_names, name)
            end
        end
    end

    _with_profiler_host(metric_names, CUPTI_PROFILER_TYPE_RANGE_PROFILER; chip) do host_ctx, config_image
        # Enable range profiler on current CUDA context
        cu_ctx = Base.unsafe_convert(CUDA.CUcontext, CUDA.context())
        enable_params = Ref(CUpti_RangeProfiler_Enable_Params(
            @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Enable_Params, pRangeProfilerObject),
            C_NULL, cu_ctx, Ptr{CUpti_RangeProfiler_Object}(0),
        ))
        cuptiRangeProfilerEnable(enable_params)
        rp_obj = enable_params[].pRangeProfilerObject

        try
            # Allocate and initialize counter data buffer
            c_names = Base.unsafe_convert.(Cstring, metric_names)
            counter_data_size = GC.@preserve metric_names c_names begin
                size_params = Ref(CUpti_RangeProfiler_GetCounterDataSize_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_GetCounterDataSize_Params, counterDataSize),
                    C_NULL, rp_obj,
                    pointer(c_names), length(c_names),
                    max_ranges, max_ranges, 0,
                ))
                cuptiRangeProfilerGetCounterDataSize(size_params)
                Int(size_params[].counterDataSize)
            end

            counter_data = Vector{UInt8}(undef, counter_data_size)
            GC.@preserve counter_data begin
                init_params = Ref(CUpti_RangeProfiler_CounterDataImage_Initialize_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, pCounterData),
                    C_NULL, rp_obj, counter_data_size, pointer(counter_data),
                ))
                cuptiRangeProfilerCounterDataImageInitialize(init_params)
            end

            # Configure and run — KernelReplay handles multi-pass internally
            GC.@preserve config_image counter_data begin
                set_params = Ref(CUpti_RangeProfiler_SetConfig_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_SetConfig_Params, targetNestingLevel),
                    C_NULL, rp_obj,
                    length(config_image), pointer(config_image),
                    counter_data_size, pointer(counter_data),
                    range_mode, CUPTI_KernelReplay,
                    max_ranges, max_nesting, 1, 0, 1,
                ))
                cuptiRangeProfilerSetConfig(set_params)
            end

            cuptiRangeProfilerStart(Ref(CUpti_RangeProfiler_Start_Params(
                @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Start_Params, pRangeProfilerObject),
                C_NULL, rp_obj,
            )))

            # Run user code with callback enabled for kernel name capture
            enable!(cb_cfg) do
                f()
            end

            cuptiRangeProfilerStop(Ref(CUpti_RangeProfiler_Stop_Params(
                @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Stop_Params, isAllPassSubmitted),
                C_NULL, rp_obj, 0, 0, 0,
            )))

            cuptiRangeProfilerDecodeData(Ref(CUpti_RangeProfiler_DecodeData_Params(
                @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_DecodeData_Params, numOfRangeDropped),
                C_NULL, rp_obj, 0,
            )))

            # Extract and evaluate results
            GC.@preserve counter_data begin
                info_params = Ref(CUpti_RangeProfiler_GetCounterDataInfo_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_GetCounterDataInfo_Params, numTotalRanges),
                    C_NULL, pointer(counter_data), counter_data_size, 0,
                ))
                cuptiRangeProfilerGetCounterDataInfo(info_params)
                num_ranges = Int(info_params[].numTotalRanges)

                range_names = String[]
                values = Matrix{Float64}(undef, num_ranges, length(metric_names))

                for i in 0:(num_ranges-1)
                    range_info = Ref(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params(
                        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, rangeName),
                        C_NULL, pointer(counter_data), counter_data_size,
                        i, Base.unsafe_convert(Cstring, "/"), Cstring(C_NULL),
                    ))
                    cuptiRangeProfilerCounterDataGetRangeInfo(range_info)
                    push!(range_names, unsafe_string(range_info[].rangeName))

                    vals = evaluate_metrics(host_ctx, counter_data, i, metric_names)
                    values[i+1, :] .= vals
                end

                return RangeProfileResult(range_names, kernel_names, metric_names, values)
            end
        finally
            cuptiRangeProfilerDisable(Ref(CUpti_RangeProfiler_Disable_Params(
                @CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Disable_Params, pRangeProfilerObject),
                C_NULL, rp_obj,
            )))
        end
    end
end


#
# PM sampling API
#

"""
    PmSample

A single PM sampling data point with timestamps and metric values.
"""
struct PmSample
    start_timestamp::UInt64
    end_timestamp::UInt64
    values::Vector{Float64}
end

"""
    PmSamplingResult

Result from PM sampling profiling.
"""
struct PmSamplingResult
    metric_names::Vector{String}
    samples::Vector{PmSample}
end

"""
    pm_sample(f, metric_names::Vector{String};
              sampling_interval=10000,
              max_samples=1024,
              hw_buffer_size=16*1024*1024) -> PmSamplingResult

Collect periodic hardware counter samples while `f()` executes.
The GPU samples counters every `sampling_interval` clock cycles.

```julia
result = CUPTI.pm_sample(["sm__cycles_active.avg", "dram__throughput.avg.pct_of_peak_sustained_elapsed"];
                          sampling_interval=5000) do
    for i in 1:100
        CUDA.@sync my_kernel(args...)
    end
end

for s in result.samples
    println("t=\$(s.start_timestamp): ", s.values)
end
```
"""
function pm_sample(f, metric_names::Vector{String};
                   chip::Union{String,Nothing}=nothing,
                   device_index::Int=0,
                   sampling_interval::UInt64=UInt64(10000),
                   max_samples::Int=1024,
                   hw_buffer_size::Int=16*1024*1024,
                   trigger_mode::CUpti_PmSampling_TriggerMode=CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL)
    # PM sampling needs a counter availability image for forward chip compatibility
    avail_image = _get_counter_availability(device_index)

    _with_profiler_host(metric_names, CUPTI_PROFILER_TYPE_PM_SAMPLING;
                        chip, counter_availability_image=avail_image) do host_ctx, config_image
        # Enable PM sampling on device
        enable_params = Ref(CUpti_PmSampling_Enable_Params(
            @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Enable_Params, pPmSamplingObject),
            C_NULL, device_index, Ptr{CUpti_PmSampling_Object}(0),
        ))
        cuptiPmSamplingEnable(enable_params)
        pm_obj = enable_params[].pPmSamplingObject

        try
            # Allocate and initialize counter data buffer
            c_names = Base.unsafe_convert.(Cstring, metric_names)
            counter_data_size = GC.@preserve metric_names c_names begin
                size_params = Ref(CUpti_PmSampling_GetCounterDataSize_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterDataSize_Params, counterDataSize),
                    C_NULL, pm_obj,
                    pointer(c_names), length(c_names),
                    max_samples, 0,
                ))
                cuptiPmSamplingGetCounterDataSize(size_params)
                Int(size_params[].counterDataSize)
            end

            counter_data = Vector{UInt8}(undef, counter_data_size)
            GC.@preserve counter_data begin
                init_params = Ref(CUpti_PmSampling_CounterDataImage_Initialize_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_CounterDataImage_Initialize_Params, pCounterData),
                    C_NULL, pm_obj, counter_data_size, pointer(counter_data),
                ))
                cuptiPmSamplingCounterDataImageInitialize(init_params)
            end

            # Configure sampling parameters
            GC.@preserve config_image begin
                cfg_params = Ref(CUpti_PmSampling_SetConfig_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_SetConfig_Params, hwBufferAppendMode),
                    C_NULL, pm_obj,
                    length(config_image), pointer(config_image),
                    hw_buffer_size, sampling_interval, trigger_mode,
                    CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST,
                ))
                cuptiPmSamplingSetConfig(cfg_params)
            end

            # Start sampling, run user code, stop
            cuptiPmSamplingStart(Ref(CUpti_PmSampling_Start_Params(
                @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Start_Params, pPmSamplingObject),
                C_NULL, pm_obj,
            )))
            try
                f()
            finally
                cuptiPmSamplingStop(Ref(CUpti_PmSampling_Stop_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Stop_Params, pPmSamplingObject),
                    C_NULL, pm_obj,
                )))
            end

            # Decode and extract samples
            GC.@preserve counter_data begin
                cuptiPmSamplingDecodeData(Ref(CUpti_PmSampling_DecodeData_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_DecodeData_Params, overflow),
                    C_NULL, pm_obj,
                    pointer(counter_data), counter_data_size,
                    CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER, 0,
                )))

                info_params = Ref(CUpti_PmSampling_GetCounterDataInfo_Params(
                    @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterDataInfo_Params, numCompletedSamples),
                    C_NULL, pointer(counter_data), counter_data_size, 0, 0, 0,
                ))
                cuptiPmSamplingGetCounterDataInfo(info_params)
                num_samples = Int(info_params[].numCompletedSamples)

                samples = PmSample[]
                for i in 0:(num_samples-1)
                    sample_info = Ref(CUpti_PmSampling_CounterData_GetSampleInfo_Params(
                        @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_CounterData_GetSampleInfo_Params, endTimestamp),
                        C_NULL, pm_obj,
                        pointer(counter_data), counter_data_size, i, 0, 0,
                    ))
                    cuptiPmSamplingCounterDataGetSampleInfo(sample_info)
                    si = sample_info[]
                    vals = evaluate_metrics(host_ctx, counter_data, i, metric_names)
                    push!(samples, PmSample(si.startTimestamp, si.endTimestamp, vals))
                end

                return PmSamplingResult(metric_names, samples)
            end
        finally
            cuptiPmSamplingDisable(Ref(CUpti_PmSampling_Disable_Params(
                @CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Disable_Params, pPmSamplingObject),
                C_NULL, pm_obj,
            )))
        end
    end
end


function Base.string(memory_kind::CUpti_ActivityMemoryKind)
    if memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN
        "unknown"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE
        "pageable"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_PINNED
        "pinned"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE
        "device"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_ARRAY
        "array"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED
        "managed"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC
        "device static"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC
        "managed static"
    else
        "unknown"
    end
end
