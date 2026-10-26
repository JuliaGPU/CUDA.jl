# Profiler control

"""
    @profile [trace=false] [raw=false] code...
    @profile external=true code...

Profile the GPU execution of `code`.

There are two modes of operation, depending on whether `external` is `true` or `false`.
The default value depends on whether Julia is being run under an external profiler.

## Integrated profiler (`external=false`, the default)

In this mode, CUDA.jl will profile the execution of `code` and display the result. By
default, a summary of host and device-side execution will be show, including any NVTX
events. To display a chronological trace of the captured activity instead, `trace` can be
set to `true`. Trace output will include an ID column that can be used to match host-side
and device-side activity. If `raw` is `true`, all data will always be included, even if it
may not be relevant.

Slow operations will be highlighted in the output: Entries colored in yellow are among the
slowest 25%, while entries colored in red are among the slowest 5% of all operations.

The results can also be written as JSON, e.g., for processing by scripts or coding agents,
with `show(stdout, "application/json", results)`. See
[`Profile.ProfileResults`](@ref CUDATools.Profile.ProfileResults) for details.

To collect hardware metrics for every kernel instead, use
[`CUDA.@metrics`](@ref CUDATools.@metrics).

!!! compat "Julia 1.9" 
    This functionality is only available on Julia 1.9 and later.

## External profilers (`external=true`, when an external profiler is detected)

For more advanced profiling, it is possible to use an external profiling tool, such as
NSight Systems or NSight Compute. When doing so, it is often advisable to only enable the
profiler for the specific code region of interest. This can be done by wrapping the code
with `CUDA.@profile external=true`, which used to be the only way to use this macro.
"""
macro profile(ex...)
    # destructure the `@profile` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # extract keyword arguments that are handled by this macro
    external = quote
        if $Profile.detect_cupti()
            @info "This Julia session is already being profiled; defaulting to the external profiler." maxlog=1 _id=:profile
            true
        else
            false
        end
    end
    remaining_kwargs = Expr[]
    for kwarg in kwargs
        if Meta.isexpr(kwarg, :(=))
            key, value = kwarg.args
            if key == :external
                isa(value, Bool) || throw(ArgumentError("Invalid value for keyword argument `external`: got `$value`, expected literal boolean value"))
                external = value
            else
                push!(remaining_kwargs, Expr(:kw, key, esc(value)))
            end
        else
            throw(ArgumentError("Invalid keyword argument to CUDA.@profile: $kwarg"))
        end
    end

    quote
        profiled_code() = $(esc(code))
        if $external
            $Profile.profile_externally(profiled_code; $(remaining_kwargs...))
        else
            $Profile.profile_internally(profiled_code; $(remaining_kwargs...))
        end
    end
end

"""
    CUDA.@bprofile [time=1.0] [kwargs...] code...

Benchmark the given code by running it for `time` seconds, and report the results using
the internal profiler `CUDA.@profile`.

The `time` keyword argument is optional, and defaults to `1.0` seconds. Other keyword
arguments are forwarded to `CUDA.@profile`.

See also: [`CUDA.@profile`](@ref).
"""
macro bprofile(ex...)
    # destructure the `@profile` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # extract keyword arguments that are handled by this macro
    remaining_kwargs = Expr[]
    for kwarg in kwargs
        if Meta.isexpr(kwarg, :(=))
            key, value = kwarg.args
            if key == :external
                error("The `external` keyword argument is not supported by `CUDA.@bprofile`")
            else
                push!(remaining_kwargs, Expr(:kw, key, esc(value)))
            end
        else
            throw(ArgumentError("Invalid keyword argument to CUDA.@bprofile: $kwarg"))
        end
    end

    quote
        benchmarked_code() = $(esc(code))
        $Profile.benchmark_and_profile(benchmarked_code; $(remaining_kwargs...))
    end
end
"""
    @metrics [metrics=:overview] [trace=false] code...

Collect hardware metrics for every kernel launched by `code`, similar to NSight Compute,
and return them as a [`Profile.KernelProfile`](@ref).

By default (`metrics=:overview`), this reports how close each kernel comes to the peak
throughput of the GPU's compute units and memory subsystems. `metrics=:memory` reports DRAM
traffic, while a vector of strings collects the given CUPTI metrics (as listed by
`ncu --query-metrics`). Displaying the result shows a summary per kernel, or every launch
when `trace` is `true`.

To collect metrics, kernels are executed one at a time, and replayed as often as needed
to read all counters (restoring memory in between). Durations and other metrics thus
describe each kernel in isolation, and the code runs considerably slower. Only kernels
launched in the current context are profiled, at most 512 of them.

The results can also be written as JSON with `show(stdout, "application/json", results)`.

See also: [`CUDA.@profile`](@ref CUDATools.@profile), which traces host and device activity.
"""
macro metrics(ex...)
    code = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        Meta.isexpr(kwarg, :(=)) ||
            throw(ArgumentError("Invalid keyword argument to CUDA.@metrics: $kwarg"))
        key, value = kwarg.args
        Expr(:kw, key, esc(value))
    end

    quote
        profiled_code() = $(esc(code))
        # like the default value of `external` in `@profile`, this activates NVTX on first
        # use, which breaks compiling kernels further down the same call on Julia 1.10.
        $Profile.detect_cupti()
        $Profile.profile_metrics(profiled_code; $(kwargs...))
    end
end
@public @profile, @bprofile, @metrics, Profile


module Profile

using CUDACore
using CUDACore: CuDim3
using CUPTI

using Crayons: @crayon_str, Crayon
using NVTX: NVTX
using PrettyTables: PrettyTables, TextHighlighter, pretty_table
using Printf: Printf, @sprintf
using Statistics: mean, quantile, std
using demumble_jll: demumble


#
# helpers for NamedTuple-of-vector tables
#

# Push a row to a NamedTuple of vectors, filling `missing` for absent keys
function push_row!(t::NamedTuple, row::NamedTuple)
    for k in keys(t)
        if haskey(row, k)
            push!(t[k], row[k])
        else
            push!(t[k], missing)
        end
    end
    return t
end

# Filter rows by a boolean mask, returns new NamedTuple
filtermask(t::NamedTuple, mask::AbstractVector{Bool}) = map(col -> col[mask], t)

# the name of an NVTX range or marker, qualified by its domain
function nvtx_name(name, domain)
    if name !== missing && domain !== missing
        "$(domain).$(name)"
    elseif name !== missing
        "$name"
    else
        missing
    end
end

# Call `f` without inferring the call, so that callers aren't invalidated along with the
# loosely-typed instance of `f` that inference would otherwise create for their arguments.
barrier(f, args...) = Base.inferencebarrier(f)(args...)

# NVTX colors are ARGB, while Crayons expects RGB.
nvtx_crayon(color::UInt32) = Crayon(; foreground=color & 0x00ffffff)

#
# external profiler
#

function profile_externally(@nospecialize(f))
    # wait for the device to become idle
    CUDACore.cuCtxSynchronize()

    start()
    try
        f()
    finally
        stop()
    end
end

const _cupti_active = Ref{Union{Nothing,Bool}}(nothing)
const _nvtx_activated = Ref{Bool}(false)

function detect_cupti()
    cached = _cupti_active[]
    cached === nothing || return cached

    # Some CUPTI versions crash while probing Tegra without profiling permissions.
    CUPTI.can_profile() || return false

    if !_nvtx_activated[]
        # If we're not running under an external profiler, let CUPTI handle NVTX events
        # We cannot run this unconditionally during initialization to avoid interfering
        # with other cupti users who may not operate via CUDA.@profile [like Reactant/XLA]
        if !NVTX.isactive() && CUPTI.version() != v"13.0.0" # NVIDIA/NVTX#125
            ENV["NVTX_INJECTION64_PATH"] = CUPTI.libcupti
            NVTX.activate()
        end
        _nvtx_activated[] = true
    end

    subscribed = try
        cfg = CUPTI.ActivityConfig([])
        CUPTI.enable!(cfg) do
            # do nothing
        end
        false
    catch err
        isa(err, CUPTIError) || rethrow()
        err.code == CUPTI.ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED || rethrow()
        true
    end
    _cupti_active[] = subscribed
end

function find_nsys()
    if haskey(ENV, "JULIA_CUDA_NSYS")
        return ENV["JULIA_CUDA_NSYS"]
    elseif haskey(ENV, "_") && contains(ENV["_"], r"nsys"i)
        # NOTE: if running as e.g. Jupyter -> nsys -> Julia, _ is `jupyter`
        return ENV["_"]
    else
        # look at a couple of environment variables that may point to NSight
        nsys = nothing
        for var in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")
            haskey(ENV, var) || continue
            for val in split(ENV[var], Sys.iswindows() ? ';' : ':')
                isfile(val) || continue
                candidate = if Sys.iswindows()
                    joinpath(dirname(val), "nsys.exe")
                else
                    joinpath(dirname(val), "nsys")
                end
                isfile(candidate) && return candidate
            end
        end
    end
    error("Running under Nsight Systems, but could not find the `nsys` binary to start the profiler. Please specify using JULIA_CUDA_NSYS=path/to/nsys, and file an issue with the contents of ENV.")
end

const __nsys = Ref{Union{Nothing,String}}()
function nsys()
    if !isassigned(__nsys)
        # find the active Nsight Systems profiler
        if haskey(ENV, "NSYS_PROFILING_SESSION_ID") && ccall(:jl_generating_output, Cint, ()) == 0
            __nsys[] = find_nsys()
        else
            __nsys[] = nothing
        end
    end

    __nsys[]
end

function nsys_sessions()
    sessions = Dict{Int,Dict{String,String}}()
    open(`$(nsys()) sessions list`, "r") do io
        header = Dict()
        for line in eachline(io)
            # parse the header
            if isempty(header)
                @assert startswith(line, r"\s+ID")
                colnames = split(line)[1:end-1] # ignore the final left-aligned column
                colranges = []
                for column in colnames
                    push!(colranges, findfirst(Regex("\\s+\\b$column\\b"), line))
                end
                for (name, range) in zip(colnames, colranges)
                    header[name] = range
                end

            # parse the data
            else
                session = Dict()
                for (name, range) in header
                    session[name] = lstrip(line[range])
                end

                id = parse(Int, session["ID"])
                delete!(session, "ID")
                sessions[id] = session
            end
        end
    end
    return sessions
end

nsys_session() = parse(Int, ENV["NSYS_PROFILING_SESSION_ID"])

nsys_state() = nsys_sessions()[nsys_session()]["STATE"]



"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
function start()
    if nsys() !== nothing
        # by default, running under NSight Systems does not activate the profiler API-based
        # ranged collection; that's done by calling `nsys start --capture-range=cudaProfilerApi`.
        # however, as of recent we cannot do this anymore when already running under the
        # capturing `nsys profile`, so we need to detect the state and act accordingly.
        try
            state = nsys_state()

            # `nsys profile`
            if state == "Collection"
                @warn """The application is already being profiled; starting the profiler is a no-op.

                         If you meant to profile a specific region, make sure to start NSight Systems in
                         delayed mode (`nsys profile --start-later=true --capture-range=cudaProfilerApi`)
                         or simply switch to the interactive `nsys launch` command."""
                return

            # `nsys profile --start-later=true`
            elseif state == "DelayedCollection"
                @error """The application is running under a delayed profiling session which CUDA.jl cannot activate.

                          If you want `CUDA.@profile` to enable the profiler, make sure
                          to pass `--capture-range=cudaProfilerApi` to `nsys profile`."""
                return

            # `nsys profile --start-later=true --capture-range=cudaProfilerApi`
            elseif state == "StartRange"

            # `nsys launch`
            elseif state == "Launched"
                run(`$(nsys()) start --capture-range=cudaProfilerApi`)

            else
                error("Unexpected state: $state")
            end
        catch err
            @error "Failed to find the active profiling session ($(nsys_session())) in the session list:\n" * read(`$(nsys()) sessions list`, String) * "\n\nPlease file an issue." exception=(err,catch_backtrace())
        end

        # it takes a while for the profiler to attach to our process
        sleep(0.01)
    end

    # actually start the capture
    CUDACore.cuProfilerStart()
end

"""
    stop()

Disables profile collection by the active profiling tool for the current context. If
profiling is already disabled, then this call has no effect.
"""
function stop()
    CUDACore.cuProfilerStop()
end


#
# integrated profiler
#

"""
    ProfileResults(...)

The results of a profiling run, as returned by [`CUDA.@profile`](@ref CUDATools.@profile).
The recommended way to interpret these results is to visualize them using the I/O stack
(e.g. by calling `display`, `print`, `string`, ...)

For programmatic access, `show(io, MIME"application/json"(), results)` writes the report
that would be displayed as JSON, with times in nanoseconds relative to the start of the
trace, and `null` for values that are not available:
- `view`: `"summary"`, or `"trace"` when `trace=true` was passed;
- `duration_ns` and `events`: the duration of the trace and the number of events in it;
- `host_time_ns`: the time the busiest thread spent calling CUDA APIs;
- `device_time_ns`: the total duration of device activity, which exceeds the duration of
  the trace when operations overlap;
- `host`, `device` and `nvtx`: arrays of API calls, device operations, and NVTX ranges.
  In summary view, each entry summarizes the events with the same `name`: the number of
  `calls`, their `total_ns` and the `time_fraction` of the trace that represents, and their
  `mean_ns`, `std_ns`, `min_ns` and `max_ns`. In trace view, each entry describes one event:
  its `start_ns` and `duration_ns`, an `id` that matches API calls with the operations they
  launched, and details like launch configurations or the size of memory operations.

It is also possible to access the fields of this struct. However, the exact format is not
guaranteed to be stable, and may change between CUDA.jl releases. Currently, it contains
three tables (NamedTuples of vectors):
- `host`, containing host-side activity;
- `device`, containing device-side activity;
- `nvtx`, with information on captured NVTX ranges and events.

See also: [`CUDA.@profile`](@ref CUDATools.@profile)
"""
Base.@kwdef struct ProfileResults
    # captured data (NamedTuples of vectors)
    host::NamedTuple
    device::NamedTuple
    nvtx::NamedTuple

    # display properties set by `@profile` kwargs
    trace::Bool=false
    raw::Bool=false
end

function profile_internally(@nospecialize(f); concurrent=true, kwargs...)
    # CUPTI 10 can crash when profiling a context that uses dynamic parallelism.
    CUPTI.version() < v"11" &&
        error("Integrated profiling requires CUPTI 11 or later; use an external profiler with older toolkits.")

    metrics_active[] &&
        error("Cannot trace activity while collecting hardware metrics.")

    if !CUPTI.can_profile()
        error("""The integrated profiler relies on CUPTI, which requires additional permissions on Tegra devices.
                 With CUDA 13 and later, grant read/write access to `/dev/nvgpu/*/prof`
                 (typically by joining its owning group) and enable non-admin profiling with
                 `NVreg_RestrictProfilingToAdminUsers=0`. With older CUDA toolkits, run Julia
                 as root. Alternatively, use an external profiler such as Nsight Systems.""")
    end

    activity_kinds = [
        # API calls
        CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
        CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME,
        # kernel execution
        concurrent ? CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL : CUPTI.CUPTI_ACTIVITY_KIND_KERNEL,
        # memory operations
        CUPTI.CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI.CUPTI_ACTIVITY_KIND_MEMSET,
        # NVTX markers
        CUPTI.CUPTI_ACTIVITY_KIND_MARKER,
    ]
    if CUPTI.version() >= v"11"
        # internal launch API records require CUDA 11 or later
        push!(activity_kinds, CUPTI.CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API)
        # NVTX marker data (payload, color, category) is optional, and CUPTI 10.2 rejects
        # enabling it with CUPTI_ERROR_NOT_COMPATIBLE (observed on Jetson)
        push!(activity_kinds, CUPTI.CUPTI_ACTIVITY_KIND_MARKER_DATA)
    end
    if CUPTI.version() >= v"11.2"
        # Check CUPTI's API version: JetPack 5's CUDA 11.4 ships an older CUPTI.
        # Enabling MEMORY2 there corrupts the stack with the CUDA 12 compat driver.
        push!(activity_kinds, CUPTI.CUPTI_ACTIVITY_KIND_MEMORY2)
    end
    cfg = CUPTI.ActivityConfig(activity_kinds)

    # wait for the device to become idle
    CUDACore.cuCtxSynchronize()

    # CUPTI timestamps count nanoseconds since an unspecified epoch (in practice, the Unix
    # epoch). converted to seconds as is, a Float64 can only resolve them to 2^-22 s
    # (~238 ns), so make them relative to the start of the session first.
    origin = Ref{UInt64}()
    CUPTI.cuptiGetTimestamp(origin)

    CUPTI.@enable! cfg begin
        # perform dummy operations to "warm up" the profiler, and avoid slow first calls.
        # we'll skip everything up until the synchronization call during processing
        CuArray([1])
        CUDACore.cuCtxSynchronize()

        f()

        # synchronize to ensure we capture all activity
        CUDACore.cuCtxSynchronize()
    end

    data = capture(cfg, origin[])
    ProfileResults(; data..., kwargs...)
end

# convert CUPTI activity records to host and device traces, with times in seconds since
# the CUPTI timestamp `origin`
function capture(cfg, origin::UInt64)
    seconds(timestamp) = (Int64(timestamp) - Int64(origin)) / 1e9

    host_trace = (
        id      = Int[],
        start   = Float64[],
        stop    = Float64[],
        name    = String[],

        tid     = Int[],
    )
    device_trace = (
        id      = Int[],
        start   = Float64[],
        stop    = Float64[],
        name    = String[],

        device  = Int[],
        context = Int[],
        stream  = Int[],

        # kernel launches
        grid            = Union{Missing,CuDim3}[],
        block           = Union{Missing,CuDim3}[],
        registers       = Union{Missing,Int64}[],
        shared_mem      = Union{Missing,@NamedTuple{static::Int64,dynamic::Int64}}[],
        local_mem       = Union{Missing,@NamedTuple{thread::Int64,total::Int64}}[],

        # memory operations
        size        = Union{Missing,Int64}[],
    )
    # lookup tables (replaces leftjoin at end)
    details = Dict{Int,String}()
    nvtx_trace = (
        id      = Int[],
        start   = Float64[],
        type    = Symbol[],
        tid     = Int[],
        name    = Union{Missing,String}[],
        domain  = Union{Missing,String}[],
    )
    nvtx_data_lookup = Dict{Int,@NamedTuple{payload::Any, color::Union{Nothing,UInt32}, category::UInt32}}()

    # memory_kind fields are sometimes typed CUpti_ActivityMemoryKind, sometimes UInt
    as_memory_kind(x) = isa(x, CUPTI.CUpti_ActivityMemoryKind) ? x : CUPTI.CUpti_ActivityMemoryKind(x)

    CUPTI.process(cfg) do ctx, stream_id, record
        # driver API calls
        if record.kind in [CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
                           CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME,
                           CUPTI.CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API]
            id = record.correlationId
            t0, t1 = seconds(record.start), seconds(record._end)

            name = if record.kind == CUPTI.CUPTI_ACTIVITY_KIND_DRIVER
                ref = Ref{Cstring}(C_NULL)
                res = CUPTI.unchecked_cuptiGetCallbackName(CUPTI.CUPTI_CB_DOMAIN_DRIVER_API,
                                                           record.cbid, ref)
                if res == CUPTI.SUCCESS
                    unsafe_string(ref[])
                elseif res == CUPTI.ERROR_INVALID_PARAMETER
                    # this can happen when using a driver that's newer than the toolkit.
                    # try to recover it from our API wrappers
                    name = string(CUPTI.CUpti_driver_api_trace_cbid_enum(record.cbid))
                    prefix = "CUPTI_DRIVER_TRACE_CBID_"
                    if startswith(name, prefix)
                        name[length(prefix)+1:end]
                    else
                        "<unknown driver API>"
                    end
                else
                    CUPTI.throw_api_error(res)
                end
            elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME
                ref = Ref{Cstring}(C_NULL)
                CUPTI.cuptiGetCallbackName(CUPTI.CUPTI_CB_DOMAIN_RUNTIME_API,
                                           record.cbid, ref)
                unsafe_string(ref[])
            else
                "<unknown activity kind>"
            end

            push_row!(host_trace, (; id, start=t0, stop=t1, name,
                                    tid=record.threadId))

        # memory operations
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MEMCPY
            id = record.correlationId
            t0, t1 = seconds(record.start), seconds(record._end)

            src_kind = as_memory_kind(record.srcKind)
            dst_kind = as_memory_kind(record.dstKind)
            name = "[copy $(string(src_kind)) to $(string(dst_kind)) memory]"

            push_row!(device_trace, (; id, start=t0, stop=t1, name,
                                      device=record.deviceId,
                                      context=record.contextId,
                                      stream=record.streamId,
                                      size=record.bytes))
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MEMSET
            id = record.correlationId
            t0, t1 = seconds(record.start), seconds(record._end)

            memory_kind = as_memory_kind(record.memoryKind)
            name = "[set $(string(memory_kind)) memory]"

            push_row!(device_trace, (; id, start=t0, stop=t1, name,
                                      device=record.deviceId,
                                      context=record.contextId,
                                      stream=record.streamId,
                                      size=record.bytes))

        # memory allocations
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MEMORY2
            # XXX: we'd prefer to postpone processing (i.e. calling format_bytes),
            #      but cannot realistically add a column for every API call

            id = record.correlationId

            memory_kind = as_memory_kind(record.memoryKind)
            str = "$(Base.format_bytes(record.bytes)), $(string(memory_kind)) memory"

            details[id] = str

        # kernel execution
        # TODO: CUPTI_ACTIVITY_KIND_CDP_KERNEL (CUpti_ActivityCdpKernel)
        elseif record.kind in [CUPTI.CUPTI_ACTIVITY_KIND_KERNEL,
                               CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL]
            id = record.correlationId
            t0, t1 = seconds(record.start), seconds(record._end)

            name = unsafe_string(record.name)
            grid = CuDim3(record.gridX, record.gridY, record.gridZ)
            block = CuDim3(record.blockX, record.blockY, record.blockZ)
            registers = record.registersPerThread
            shared_mem = (static=Int64(record.staticSharedMemory),
                          dynamic=Int64(record.dynamicSharedMemory))
            local_mem = (thread=Int64(record.localMemoryPerThread),
                         total=Int64(record.localMemoryTotal))

            push_row!(device_trace, (; id, start=t0, stop=t1, name,
                                      device=record.deviceId,
                                      context=record.contextId,
                                      stream=record.streamId,
                                      grid, block, registers,
                                      shared_mem, local_mem))

        # NVTX markers
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MARKER
            start = seconds(record.timestamp)
            name = record.name == C_NULL ? missing : unsafe_string(record.name)
            domain = record.domain == C_NULL ? missing : unsafe_string(record.domain)

            if record.flags == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS
                @assert record.objectKind == CUPTI.CUPTI_ACTIVITY_OBJECT_THREAD
                tid = record.objectId.pt.threadId
                push_row!(nvtx_trace, (; id=record.id, start, tid, type=:instant, name, domain))
            elseif record.flags == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_START
                @assert record.objectKind == CUPTI.CUPTI_ACTIVITY_OBJECT_THREAD
                tid = record.objectId.pt.threadId
                push_row!(nvtx_trace, (; id=record.id, start, tid, type=:start, name, domain))
            elseif record.flags == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_END
                @assert record.objectKind == CUPTI.CUPTI_ACTIVITY_OBJECT_THREAD
                tid = record.objectId.pt.threadId
                push_row!(nvtx_trace, (; id=record.id, start, tid, type=:end, name, domain))
            else
                @error "Unexpected NVTX marker kind $(Int(record.flags)). Please file an issue."
            end
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MARKER_DATA
            payload_accessors = Dict(
                CUPTI.CUPTI_METRIC_VALUE_KIND_DOUBLE => :metricValueDouble,
                CUPTI.CUPTI_METRIC_VALUE_KIND_UINT64 => :metricValueUint64,
                CUPTI.CUPTI_METRIC_VALUE_KIND_PERCENT => :metricValueInt64,
                CUPTI.CUPTI_METRIC_VALUE_KIND_THROUGHPUT => :metricValuePercent,
                CUPTI.CUPTI_METRIC_VALUE_KIND_INT64 => :metricValueThroughput,
                CUPTI.CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL => :metricValueUtilizationLevel
            )
            payload = if haskey(payload_accessors, record.payloadKind)
                getproperty(record.payload, payload_accessors[record.payloadKind])
            else
                @error "Unexpected CUPTI metric kind $(Int(record.payloadKind)). Please file an issue."
                nothing
            end

            color = if record.flags & CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB
                record.color
            else
                nothing
            end

            nvtx_data_lookup[record.id] = (; payload, color, category=record.category)
        else
            @error "Unexpected CUPTI activity kind $(Int(record.kind)). Please file an issue."
        end
    end

    # demangle all kernel names in a single demumble invocation, which is much faster
    # than demangling them one by one.
    demangle_names!(device_trace.name)

    # add details column via Dict lookup (replaces leftjoin)
    host_details = Union{Missing,String}[get(details, id, missing) for id in host_trace.id]
    host = merge(host_trace, (; details=host_details))

    dev_details = Union{Missing,String}[get(details, id, missing) for id in device_trace.id]
    device = merge(device_trace, (; details=dev_details))

    # add NVTX data columns via Dict lookup (replaces leftjoin)
    n_nvtx = length(nvtx_trace.id)
    nvtx_payload = Vector{Any}(missing, n_nvtx)
    nvtx_color = Vector{Union{Nothing,UInt32}}(nothing, n_nvtx)
    nvtx_category = Vector{Union{Missing, UInt32}}(missing, n_nvtx)
    for i in 1:n_nvtx
        data = get(nvtx_data_lookup, nvtx_trace.id[i], nothing)
        if !isnothing(data)
            nvtx_payload[i] = data.payload
            nvtx_color[i] = data.color
            nvtx_category[i] = data.category
        end
    end
    nvtx = merge(nvtx_trace, (; payload=nvtx_payload, color=nvtx_color, category=nvtx_category))

    return (; host, device, nvtx)
end

Base.show(io::IO, results::ProfileResults) = show_report(io, results)

function show_report(io::IO, report)
    crop = if get(io, :is_pluto, false) || get(io, :jupyter, false)
        # Pluto.jl and IJulia.jl both indicate they want to limit output,
        # but they have scrollbars, so let's ignore that
        :none
    elseif io isa Base.TTY || get(io, :limit, false)::Bool
        # crop horizonally to fit the terminal
        :horizontal
    else
        :none
    end

    # render into a buffer, so that the code below (which takes long to compile) does not
    # need to be specialized on, and precompiled for, every kind of output stream
    buf = IOContext(IOContext(IOBuffer(), io), :color => get(io, :color, false)::Bool,
                    :displaysize => displaysize(io))
    show_results(buf, report, crop)
    str = String(take!(buf.io))
    if crop == :none
        # tables pad every cell to the width of their column, which for kernel names can
        # be hundreds of characters (keeping any escape sequences that follow)
        str = replace(str, r" +((?:\e\[[0-9;]*m)*)$"m => s"\1")
    end
    write(io, str)
    return
end

# without cropping, there's no right border, so that rows aren't padded to the widest one
const uncropped_format = PrettyTables.TextTableFormat(;
    horizontal_line_at_beginning=false, horizontal_line_after_data_rows=false,
    vertical_line_at_beginning=false, vertical_line_after_data_columns=false)

# print the `columns` of a table. PrettyTables gets a matrix with their values, so that its
# code doesn't need to be compiled for every combination of columns (which depends on the
# data), and `formatter(value, column)` and the highlighters operate on that matrix.
function print_table(io::IOContext{IOBuffer}, crop::Symbol, df::NamedTuple,
                     columns::Vector{Symbol}; kwargs...)
    data = Matrix{Any}(undef, length(first(df)), length(columns))
    for (j, column) in enumerate(columns)
        set_column!(data, j, df[column])
    end
    print_table(io, crop, data, columns; kwargs...)
end
# called with dynamic dispatch, as the type of a column isn't known statically, which (unlike
# for Base's `setindex!`) makes the precompiled specializations available.
set_column!(data::Matrix{Any}, j::Int, column::AbstractVector) = (data[:, j] = column)
function print_table(io::IOContext{IOBuffer}, crop::Symbol, data::Matrix{Any},
                     columns::Vector{Symbol}; labels::Dict{Symbol,String},
                     alignment::Vector{Symbol}, formatter::Function,
                     highlighters::Vector{TextHighlighter}=TextHighlighter[])
    pretty_table(io, data; column_labels=[labels[column] for column in columns], alignment,
                 formatters=[(v, i, j) -> formatter(v, columns[j])], highlighters,
                 table_format=(crop == :none ? uncropped_format : PrettyTables.TextTableFormat()),
                 fit_table_in_display_horizontally=(crop == :horizontal),
                 fit_table_in_display_vertically=false)
end

# The report that displaying `results` shows: the activity in the profiled region (or the
# entire trace in raw trace mode) with times relative to its start, as tables listing every
# event (`trace=true`) or summarizing the events per name.
#
# The fields of `ProfileResults` are not concretely typed, so this and the code using its
# result are called through function barriers. Otherwise, many calls would be inferred
# loosely enough to be invalidated by packages extending Base (e.g. GPUArrays' `sortperm`),
# discarding the precompiled display code.
report(results::ProfileResults) =
    barrier(report, results.host, results.device, results.nvtx, results.trace, results.raw)
function report(host::NamedTuple, device::NamedTuple, nvtx::NamedTuple, trace::Bool, raw::Bool)
    # the columns are modified in place below. their elements are immutable, so copying
    # them suffices (and unlike `deepcopy`, is inferred)
    host, device, nvtx = map(copy, host), map(copy, device), map(copy, nvtx)

    # find the relevant part of the trace (marked by calls to 'cuCtxSynchronize')
    syncs = host.name .== "cuCtxSynchronize"
    trace_first_sync = findfirst(syncs)
    trace_first_sync === nothing && error("Could not find the start of the profiling data.")
    trace_last_sync = findlast(syncs)
    trace_first_sync == trace_last_sync && error("Could not find the end of the profiling data.")
    ## truncate the trace
    if !raw || !trace
        trace_begin = host.stop[trace_first_sync]
        trace_end = host.stop[trace_last_sync]

        first_id = host.id[trace_first_sync + 1]
        last_id = host.id[trace_last_sync - 1]
        host = filtermask(host, first_id .<= host.id .<= last_id)
        device = filtermask(device, first_id .<= device.id .<= last_id)
    else
        # in raw mode, we display the entire trace. note that we only do so when tracing,
        # because otherwise the summary would be skewed by the expensive initial API call
        # used to sink the profiler overhead.

        # inclusive bounds
        trace_begin = host.start[begin]
        trace_end = host.stop[end]
    end
    trace_time = trace_end - trace_begin

    # compute event and trace duration
    host = merge(host, (; time=host.stop .- host.start))
    device = merge(device, (; time=device.stop .- device.start))
    events = length(host.id) + length(device.id)

    # make some numbers more friendly to read
    ## make timestamps relative to the start
    host.start .-= trace_begin
    host.stop .-= trace_begin
    device.start .-= trace_begin
    device.stop .-= trace_begin
    nvtx.start .-= trace_begin
    if !raw
        # renumber event IDs from 1
        first_id = minimum([host.id; device.id])
        for df in (host, device)
            df.id .-= first_id - 1
        end

        # renumber thread IDs from 1
        threads = unique([host.tid; nvtx.tid])
        for df in (host, nvtx)
            broadcast!(df.tid, df.tid) do tid
                findfirst(isequal(tid), threads)
            end
        end
    end

    function summarize_trace(df)
        # group times by name (which can be missing for NVTX ranges)
        groups = Dict{Union{Missing,String},Vector{Float64}}()
        for (name, t) in zip(df.name, df.time)
            push!(get!(Vector{Float64}, groups, name), t)
        end

        n = length(groups)
        out_name = Vector{Union{Missing,String}}(undef, n)
        out_time = Vector{Float64}(undef, n)
        out_calls = Vector{Int}(undef, n)
        out_dist = Vector{Union{Missing,@NamedTuple{std::Float64, mean::Float64, min::Float64, max::Float64}}}(undef, n)
        for (i, (name, times)) in enumerate(groups)
            out_name[i] = name
            out_time[i] = sum(times)
            out_calls[i] = length(times)
            out_dist[i] = if length(times) == 1
                missing
            else
                (; std=std(times), mean=mean(times), min=minimum(times), max=maximum(times))
            end
        end
        out_ratio = out_time ./ trace_time

        # sort by time ratio (descending)
        perm = sortperm(out_ratio; rev=true)
        return (
            name      = out_name[perm],
            time      = out_time[perm],
            calls     = out_calls[perm],
            time_dist = out_dist[perm],
            time_ratio = out_ratio[perm],
        )
    end

    # host-side activity
    ## to determine the time the host was active, we should look at threads separately
    thread_times = Dict{Int,Float64}()
    for (tid, t) in zip(host.tid, host.time)
        thread_times[tid] = get(thread_times, tid, 0.0) + t
    end
    host_time = maximum(values(thread_times))

    ## get rid of API call version suffixes
    host.name .= replace.(host.name, r"_v\d+$" => "")

    if !raw
        # filter spammy API calls
        spammy = Set([# context and stream queries we use for nonblocking sync
                      "cuCtxGetCurrent", "cuCtxGetId", "cuCtxGetApiVersion",
                      "cuStreamQuery", "cuStreamGetId",
                      # occupancy API, done before every kernel launch
                      "cuOccupancyMaxPotentialBlockSize",
                      # driver pointer set-up
                      "cuGetProcAddress",
                      # called a lot during compilation
                      "cuDeviceGetAttribute",
                      # done before every memory operation
                      "cuPointerGetAttribute", "cuDeviceGetMemPool",
                      "cuStreamGetCaptureInfo"])
        host = filtermask(host, [name ∉ spammy for name in host.name])
    end

    if trace
        # instantaneous NVTX markers are added to the API trace. this is done even if there
        # are none, so that the type of the trace doesn't depend on the data.
        instant_mask = nvtx.type .== :instant
        n_markers = count(instant_mask)
        marker_details = Union{Missing,String}[nvtx_name(name, domain) for (name, domain)
                                               in zip(nvtx.name[instant_mask], nvtx.domain[instant_mask])]

        # append markers to host trace (with type widening for id/stop)
        host = (
            id      = vcat(Vector{Union{Missing,Int}}(host.id), fill(missing, n_markers)),
            start   = vcat(host.start, nvtx.start[instant_mask]),
            stop    = vcat(Vector{Union{Missing,Float64}}(host.stop), fill(missing, n_markers)),
            name    = vcat(host.name, fill("NVTX marker", n_markers)),
            tid     = vcat(host.tid, nvtx.tid[instant_mask]),
            details = vcat(host.details, marker_details),
            time    = vcat(host.time, zeros(Float64, n_markers)),
        )

        # sort by start time
        order = sortperm(host.start)
        host = map(col -> col[order], host)
    else
        host = summarize_trace(host)
    end

    # device-side activity
    device_time = sum(device.time)

    ## add memory throughput information. CUPTI's timestamp resolution
    ## can round very short events down to 0 ns, so guard against the
    ## resulting Inf throughput (which would later trip up format_bytes).
    throughput = Union{Missing,Float64}[
        (s === missing || !isfinite(t) || t == 0) ? missing : s / t
        for (s, t) in zip(device.size, device.time)]
    device = merge(device, (; throughput))

    if !trace
        device = summarize_trace(device)
    end

    # NVTX ranges
    # TODO: do we also want to repeat the host/device summary for each NVTX range?
    #       that's what nvprof used to do, but it's a little verbose...

    ## build lookup from end event id → stop time
    end_times = Dict{Int,Float64}()
    for i in eachindex(nvtx.id)
        if nvtx.type[i] == :end
            end_times[nvtx.id[i]] = nvtx.start[i]
        end
    end

    ## add stop time from matching end events
    starts = filtermask(nvtx, nvtx.type .== :start)
    stop_times = Float64[get(end_times, id, NaN) for id in starts.id]
    nvtx_ranges = merge(starts, (; stop=stop_times, time=stop_times .- starts.start))

    if !trace
        # merge the domain and name into a single column
        nvtx_ranges.name .= nvtx_name.(nvtx_ranges.name, nvtx_ranges.domain)

        nvtx_ranges = summarize_trace(nvtx_ranges)
    end

    return (; time=trace_time, events, host_time, device_time,
              host, device, nvtx=nvtx_ranges)
end

show_results(io::IOContext{IOBuffer}, results::ProfileResults, crop::Symbol) =
    barrier(show_activity, io, report(results), results.trace, results.raw, crop)

function show_activity(io::IOContext{IOBuffer}, r::NamedTuple, trace::Bool, raw::Bool,
                       crop::Symbol)
    println(io, "Profiler ran for $(format_time(r.time)), capturing $(r.events) events.")

    # helper function to visualize slow trace entries
    function time_highlighters(times::Vector{Float64}, columns::Vector{Symbol})
        ## filter out entries that execute _very_ quickly (like calls to cuCtxGetCurrent)
        relevant_times = times[times .>= 1e-8]

        isempty(relevant_times) && return TextHighlighter[]
        p75 = quantile(relevant_times, 0.75)
        p95 = quantile(relevant_times, 0.95)

        time = findfirst(==(:time), columns)
        TextHighlighter[
            TextHighlighter((data, i, j) -> j == time && data[i, j]::Float64 >= p95, crayon"red"),
            TextHighlighter((data, i, j) -> j == time && data[i, j]::Float64 >= p75, crayon"yellow"),
            TextHighlighter((data, i, j) -> columns[j] == :name && data[i, time]::Float64 >= p75,
                            crayon"bold")]
    end

    trace_column_names = Dict(
        :id            => "ID",
        :start         => "Start",
        :time          => "Time",
        :grid          => "Blocks",
        :tid           => "Thread",
        :block         => "Threads",
        :registers     => "Regs",
        :shared_mem    => "Shared Mem",
        :local_mem     => "Local Mem",
        :size          => "Size",
        :throughput    => "Throughput",
        :device        => "Device",
        :stream        => "Stream",
        :name          => "Name",
        :domain        => "Domain",
        :details       => "Details",
        :payload       => "Payload",
    )

    summary_column_names = Dict(
        :time          => "Total time",
        :time_ratio    => "Time (%)",
        :calls         => "Calls",
        :time_dist     => "Time distribution",
        :name          => "Name",
    )

    function trace_formatter(v, column)
        if v === missing
            "-"
        elseif column in (:start, :time)
            format_time(v)
        elseif column == :size
            Base.format_bytes(v)
        elseif column == :shared_mem
            if raw || v.static > 0 && v.dynamic > 0
                "$(Base.format_bytes(v.static)) static, $(Base.format_bytes(v.dynamic)) dynamic"
            elseif v.static > 0
                "$(Base.format_bytes(v.static)) static"
            elseif v.dynamic > 0
                "$(Base.format_bytes(v.dynamic)) dynamic"
            else
                "-"
            end
        elseif column == :local_mem
            "$(Base.format_bytes(v.thread)) / $(Base.format_bytes(v.total))"
        elseif column == :throughput
            Base.format_bytes(v) * "/s"
        elseif column == :device
            name(CuDevice(v))
        elseif v isa CuDim3
            format_dim(v)
        else
            v
        end
    end

    function summary_formatter(v, column)
        if column == :name && v === missing
            "-"
        elseif column == :time_ratio
            format_percentage(v)
        elseif column == :time
            format_time(v)
        elseif column == :time_dist
            if v === missing
                ""
            else
                mean, std, min, max = format_time(v.mean, v.std, v.min, v.max)
                @sprintf("%9s ± %-6s (%6s ‥ %s)", mean, std, min, max)
            end
        else
            v
        end
    end

    function print_summary(df)
        columns = [:time_ratio, :time, :calls]
        if any(!ismissing, df.time_dist)
            push!(columns, :time_dist)
        end
        push!(columns, :name)

        alignment = [k in (:name, :time_dist) ? :l : :r for k in columns]
        print_table(io, crop, df, columns; labels=summary_column_names, alignment,
                    formatter=summary_formatter,
                    highlighters=time_highlighters(df.time, columns))
    end

    function print_trace(df, columns; highlighters=TextHighlighter[])
        alignment = [k == :name ? :l : :r for k in columns]
        print_table(io, crop, df, columns; labels=trace_column_names, alignment,
                    formatter=trace_formatter,
                    highlighters=[highlighters; time_highlighters(df.time, columns)])
    end

    # whether a column contains useful information
    varies(column) = raw || length(unique(column)) > 1
    present(column) = raw || any(!ismissing, column)

    # host-side activity
    let df = r.host
        if !isempty(df.name)
            host_ratio = r.host_time / r.time
            println(io, "\nHost-side activity: calling CUDA APIs took $(format_time(r.host_time)) ($(format_percentage(host_ratio)) of the trace)")
        end
        if isempty(df.name)
            println(io, "\nNo host-side activity was recorded.")
        elseif trace
            columns = [:id, :start, :time]
            varies(df.tid) && push!(columns, :tid)
            push!(columns, :name)
            any(!ismissing, df.details) && push!(columns, :details)
            print_trace(df, columns)
        else
            print_summary(df)
        end
    end

    # device-side activity
    let df = r.device
        if !isempty(df.name)
            device_ratio = r.device_time / r.time
            println(io, "\nDevice-side activity: GPU was busy for $(format_time(r.device_time)) ($(format_percentage(device_ratio)) of the trace)")
        end

        if isempty(df.name)
            println(io, "\nNo device-side activity was recorded.")
        elseif trace
            columns = [:id, :start, :time]
            ## device/stream identification
            varies(df.device) && push!(columns, :device)
            varies(df.stream) && push!(columns, :stream)
            ## kernel details (can be missing)
            present(df.block) && push!(columns, :block)
            present(df.grid) && push!(columns, :grid)
            present(df.registers) && push!(columns, :registers)
            if raw || any(val->!ismissing(val) && (val.static > 0 || val.dynamic > 0), df.shared_mem)
                push!(columns, :shared_mem)
            end
            if raw || any(val->!ismissing(val) && val.thread > 0, df.local_mem)
                push!(columns, :local_mem)
            end
            ## memory details (can be missing)
            if present(df.size)
                push!(columns, :size)
                push!(columns, :throughput)
            end
            push!(columns, :name)
            print_trace(df, columns)
        else
            print_summary(df)
        end
    end

    # NVTX ranges
    let df = r.nvtx
        isempty(df.name) && return
        println(io, "\nNVTX ranges:")

        if trace
            columns = [:id, :start, :time]
            varies(df.tid) && push!(columns, :tid)
            present(df.domain) && push!(columns, :domain)
            present(df.name) && push!(columns, :name)
            present(df.payload) && push!(columns, :payload)

            # use color information as provided by NVTX
            id = findfirst(==(:id), columns)
            color_highlighters = TextHighlighter[]
            for color in unique(df.color)
                if color !== nothing
                    ids = Set(df.id[isequal.(df.color, color)])
                    highlighter = TextHighlighter(nvtx_crayon(color)) do data, i, j
                        columns[j] in (:name, :domain) && data[i, id] in ids
                    end
                    push!(color_highlighters, highlighter)
                end
            end

            print_trace(df, columns; highlighters=color_highlighters)
        else
            print_summary(df)
        end
    end

    return
end

format_percentage(x::Number) = @sprintf("%.2f%%", x * 100)

function format_dim(v::CuDim3)
    if v.z != 1
        "$(Int(v.x))×$(Int(v.y))×$(Int(v.z))"
    elseif v.y != 1
        "$(Int(v.x))×$(Int(v.y))"
    else
        "$(Int(v.x))"
    end
end

function format_time(ts::Number...)
    # the first number determines the scale and unit
    t = ts[1]
    range, unit = if abs(t) < 1e-6  # less than 1 microsecond
        1e9, "ns"
    elseif abs(t) < 1e-3  # less than 1 millisecond
        1e6, "µs"
    elseif abs(t) < 1  # less than 1 second
        1e3, "ms"
    else
        1, "s"
    end

    strs = String[]

    # only the first number displays the unit
    let io = IOBuffer()
        Base.print(io, round(t * range, digits=2), " ", unit)
        push!(strs, String(take!(io)))
    end

    # the other numbers are simply scaled
    for t in ts[2:end]
        let io = IOBuffer()
            Base.print(io, round(t * range, digits=2))
            push!(strs, String(take!(io)))
        end
    end

    if length(strs) == 1
        return strs[1]
    else
        return strs
    end
end


#
# kernel metrics
#

# how to collect and display a metric
struct MetricColumn
    name::String        # column header
    metric::String      # CUPTI metric name
    kind::Symbol        # :time (in ns), :percent, :bytes, or :number
end

const duration_metric = MetricColumn("Time", "gpu__time_duration.sum", :time)

# the throughput of compute and memory units, relative to their peak, as in NSight
# Compute's "speed of light" section. L1 is relative to the cycles the SMs were active,
# the others relative to the kernel's duration.
const overview_metrics = [
    duration_metric,
    MetricColumn("Compute", "sm__throughput.avg.pct_of_peak_sustained_elapsed", :percent),
    MetricColumn("Memory", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", :percent),
    MetricColumn("DRAM", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", :percent),
    MetricColumn("L2", "lts__throughput.avg.pct_of_peak_sustained_elapsed", :percent),
    MetricColumn("L1/TEX", "l1tex__throughput.avg.pct_of_peak_sustained_active", :percent),
]

const memory_metrics = [
    duration_metric,
    MetricColumn("DRAM", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", :percent),
    MetricColumn("DRAM traffic", "dram__bytes.sum", :bytes),
]

function metric_columns(metrics::Symbol)
    metrics == :overview && return overview_metrics
    metrics == :memory && return memory_metrics
    throw(ArgumentError("Unknown metric preset `:$metrics`; use `:overview` or `:memory`, or a vector of metric names"))
end
function metric_columns(metrics::AbstractVector{<:AbstractString})
    isempty(metrics) && throw(ArgumentError("No metrics to collect"))
    map(metrics) do metric
        kind = contains(metric, ".pct") ? :percent : :number
        MetricColumn(metric, metric, kind)
    end
end
metric_columns(metrics) =
    throw(ArgumentError("Invalid value for `metrics`: expected `:overview`, `:memory`, or a vector of metric names, got $(repr(metrics))"))

"""
    KernelProfile

The hardware metrics collected by [`CUDA.@metrics`](@ref CUDATools.@metrics). Displaying
this object shows a summary per kernel, or every launch when `trace=true` was passed.

For programmatic access, `show(io, MIME"application/json"(), p)` writes the report that
would be displayed as JSON:
- `view`: `"summary"`, or `"trace"` when `trace=true` was passed;
- `device`, and the number of kernel `launches` and of replay `passes`;
- `metrics`: the collected metrics, with their CUPTI `name`, the `label` used when
  displaying them, and their `unit` (`null` if unknown);
- `kernels`: in summary view, every kernel with its number of `calls` and the
  `time_fraction` they took, sorted by duration; in trace view, every launch with its `id`,
  `grid` and `block` size, `registers` per thread, and `stream`. Each entry has a `metrics`
  object with the value of every metric, averaged over the launches of a kernel in
  summary view.

The following fields are also available:
- `kernels`: a table (NamedTuple of vectors) with one row per kernel launch, in the order
  they executed: the launch's correlation `id`, kernel `name`, `grid` and `block`
  dimensions, `registers` per thread, and `stream`;
- `metrics`: the names of the CUPTI metrics that were collected;
- `values`: a matrix with, for every launch (row), the value of every metric (column).
  Durations are in nanoseconds, percentages between 0 and 100;
- `passes`: how often every kernel was executed to collect these metrics.

See also: [`CUDA.@metrics`](@ref CUDATools.@metrics)
"""
Base.@kwdef struct KernelProfile
    device::String
    passes::Int

    kernels::NamedTuple
    metrics::Vector{String}
    values::Matrix{Float64}

    # display properties
    columns::Vector{MetricColumn}
    trace::Bool=false
end

# the maximal number of kernel launches to collect metrics for. CUPTI's buffer needs
# space for every launch up front (up to 400 KiB each for the overview metrics).
const max_launches = 512

# CUPTI versions whose range profiler doesn't work: 12.8 and 12.9 don't profile any kernel,
# while 13.0 and 13.1 don't record when kernels executed, so we can't attribute ranges to
# launches (observed on Turing, Ampere and Blackwell).
broken_range_profiler() = v"12.8" <= CUPTI.version() < v"13.2"

# CUPTI may also fail silently on configurations we haven't tested. to detect that, every
# session ends with this kernel, which loads without compiling Julia code.
const probe_ptx = """
    .version 6.0
    .target sm_50
    .address_size 64
    .visible .entry cuda_jl_metrics_probe() { ret; }
    """

# set while collecting metrics, during which the activity lock is held
const metrics_active = Threads.Atomic{Bool}(false)

function profile_metrics(@nospecialize(f); metrics=:overview, trace::Bool=false)
    columns = metric_columns(metrics)
    check_metrics_support()

    # claim CUPTI for the whole session without waiting, as the code being profiled may be
    # waiting for whoever holds it. checking `islocked` first rejects nesting in a session
    # of this task, which the reentrant `trylock` would admit.
    if islocked(CUPTI.activity_lock) || !trylock(CUPTI.activity_lock)
        error("Cannot collect hardware metrics while another profiling session is active.")
    end
    metrics_active[] = true
    try
        collect_metrics(f, metrics, columns, trace)
    finally
        metrics_active[] = false
        unlock(CUPTI.activity_lock)
    end
end

function collect_metrics(@nospecialize(f), metrics, columns, trace)
    ctx = context()
    CUPTI.check_profiler_support(device(ctx))
    probe = CuFunction(CuModule(probe_ptx), "cuda_jl_metrics_probe")
    activity = CUPTI.ActivityConfig([CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL])
    CUDACore.cuCtxSynchronize()
    profile = try
        if !(metrics isa AbstractVector)
            # not every GPU supports all metrics of a preset, e.g., integrated GPUs don't
            # report DRAM throughput
            columns = CUPTI.ProfilerHost(ctx) do host
                filter(col -> is_available(host, col.metric), columns)
            end
        end

        # the activity API identifies the launches, which are serialized, in the order
        # the profiler numbers them. the profiler is set up before running `f`, so that
        # configuration errors (e.g. unknown metrics) don't execute the user's code.
        CUPTI.@enable! activity begin
            CUPTI.range_profile([col.metric for col in columns]; ctx,
                                max_ranges=max_launches+1) do
                f()
                # the code may have left another context current
                context!(ctx) do
                    CUDACore.cuCtxSynchronize()
                    cudacall(probe, Tuple{})
                    CUDACore.cuCtxSynchronize()
                end
            end
        end
    catch err
        if err isa CUPTIError && err.code == CUPTI.CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
            error("""Collecting hardware metrics requires access to the GPU's performance counters,
                     which the NVIDIA driver restricts to administrators by default. Either run Julia as
                     root, or allow all users by loading the `nvidia` kernel module with
                     `NVreg_RestrictProfilingToAdminUsers=0`.
                     See https://developer.nvidia.com/ERR_NVGPUCTRPERM for details.""")
        elseif err isa CUPTIError && err.code == CUPTI.CUPTI_ERROR_HARDWARE_BUSY
            error("Cannot collect hardware metrics, as the GPU is being profiled by another process.")
        end
        rethrow()
    end

    kernels = kernel_launches(activity, ctx)
    nlaunches = length(kernels.id)
    nranges = size(profile.values, 1)
    if nranges == 0 || isempty(kernels.name) || kernels.name[end] != "cuda_jl_metrics_probe"
        error("""CUPTI $(CUPTI.version()) did not collect hardware metrics on this system, without reporting an error.
                 Try a more recent version of CUDA.""")
    elseif nlaunches > max_launches + 1
        error("Hardware metrics can be collected for at most $max_launches kernel launches, but $(nlaunches-1) kernels were launched. Profile a smaller region of code.")
    elseif nlaunches != nranges
        error("Could not attribute hardware metrics to kernels: CUPTI reported $nranges ranges for $nlaunches kernel launches. Please file an issue.")
    end

    # drop the probe
    kernels = map(col -> col[1:end-1], kernels)
    values = profile.values[1:end-1, :]
    KernelProfile(; device=name(device(ctx)), passes=profile.passes, kernels,
                    metrics=[col.metric for col in columns], values, columns, trace)
end

function is_available(host, metric)
    try
        CUPTI.metric_properties(host, metric)
        return true
    catch err
        err isa CUPTIError && err.code == CUPTI.CUPTI_ERROR_INVALID_METRIC_NAME || rethrow()
        return false
    end
end

function check_metrics_support()
    if !CUPTI.can_profile()
        error("""Collecting hardware metrics relies on CUPTI, which requires additional permissions on Tegra devices.
                 See the documentation of `CUDA.@profile` for details.""")
    end
    if !CUPTI.has_range_profiler()
        error("Collecting hardware metrics requires CUPTI from CUDA 12.6 Update 2 or later (CUPTI $(CUPTI.version()) is loaded).")
    end
    if broken_range_profiler()
        error("CUPTI $(CUPTI.version()) cannot collect hardware metrics. Use CUDA 13.2 or later.")
    end
    if detect_cupti()
        error("This Julia session is already being profiled by an external tool, so CUDA.jl cannot collect hardware metrics.")
    end
end

# the kernel launches in `ctx`, in the order they executed
function kernel_launches(cfg, ctx)
    ctx_id = Ref{UInt32}()
    CUPTI.cuptiGetContextId(ctx, ctx_id)

    records = []
    CUPTI.process(cfg) do _, _, record
        record.contextId == ctx_id[] && push!(records, record)
    end

    # CUPTI launches a helper kernel after every profiled one, which it doesn't profile.
    # its correlation ID is either the profiled kernel's or a new one, so identify it by
    # name and launch configuration. anything unexpected fails the range count check.
    filter!(!is_profiler_kernel, records)

    # CUPTI doesn't record when kernels in a graph executed, so we can't tell which range
    # they correspond to
    any(record -> record.graphId != 0, records) &&
        error("Hardware metrics cannot be collected for kernels launched by CUDA graphs. Launch them outside of a graph instead.")
    any(record -> record.start == 0, records) &&
        error("CUPTI $(CUPTI.version()) did not record when kernels executed, so hardware metrics cannot be attributed to them. Try a more recent version of CUDA.")
    sort!(records; by=record->record.start)
    for i in 2:length(records)
        records[i].start == records[i-1].start &&
            error("Could not attribute hardware metrics to kernels: launches started at the same time. Please file an issue.")
    end

    names = String[unsafe_string(record.name) for record in records]
    demangle_names!(names)
    (; id = Int[record.correlationId for record in records],
       name = names,
       grid = CuDim3[CuDim3(record.gridX, record.gridY, record.gridZ) for record in records],
       block = CuDim3[CuDim3(record.blockX, record.blockY, record.blockZ) for record in records],
       registers = Int[record.registersPerThread for record in records],
       stream = Int[record.streamId for record in records])
end

is_profiler_kernel(record) =
    unsafe_string(record.name) == "WaitNs" &&
    (record.gridX, record.gridY, record.gridZ) == (1, 1, 1) &&
    (record.blockX, record.blockY, record.blockZ) == (1, 1, 1)

Base.show(io::IO, p::KernelProfile) = show_report(io, p)

# The report that displaying `p` shows: a table of kernel launches (`trace=true`) or of
# kernels sorted by their total duration, if known, and a matrix with the value of every
# metric for each row (averaged over the launches of each kernel).
report(p::KernelProfile) = barrier(report, p.kernels, p.values, p.columns, p.trace)
function report(kernels::NamedTuple, values::Matrix{Float64}, columns::Vector{MetricColumn},
                trace::Bool)
    timecol = findfirst(col -> col.metric == duration_metric.metric, columns)

    trace && return (; rows=kernels, values)

    groups = Dict{String,Vector{Int}}()
    order = String[]
    for (i, name) in enumerate(kernels.name)
        haskey(groups, name) || push!(order, name)
        push!(get!(Vector{Int}, groups, name), i)
    end
    if timecol !== nothing
        sort!(order; by=name->sum(values[groups[name], timecol]), rev=true)
    end
    total_time = timecol === nothing ? 0.0 : sum(values[:, timecol])
    rows = (; time_ratio = Float64[timecol === nothing ? NaN :
                                   sum(values[groups[name], timecol]) / total_time
                                   for name in order],
              calls = Int[length(groups[name]) for name in order],
              name = order)
    means = Float64[mean(values[groups[name], j]) for name in order, j in axes(values, 2)]
    return (; rows, values=means)
end

function show_results(io::IOContext{IOBuffer}, p::KernelProfile, crop::Symbol)
    nlaunches = length(p.kernels.id)
    if nlaunches == 0
        println(io, "No kernels were launched.")
        return
    end
    launches = nlaunches == 1 ? "1 kernel launch" : "$nlaunches kernel launches"
    passes = p.passes == 1 ? "" : ", executing each kernel $(p.passes) times"
    println(io, "Collected hardware metrics for $launches on $(p.device)$passes.")
    println(io, "Kernels were serialized, so these metrics describe each kernel in isolation.\n")

    (; rows, values) = report(p)
    barrier(show_metrics, io, rows, values, p.columns, p.trace, crop)

    if any(col -> col.kind == :percent && col in overview_metrics, p.columns)
        l1 = any(col -> col.name == "L1/TEX", p.columns) ?
            " (or the cycles the SMs were active, for L1/TEX)" : ""
        println(io, """

            Throughput is relative to the peak sustained rate of each unit during the kernel's
            execution$l1.""")
        get(io, :color, false)::Bool && println(io, "Values of at least 80% are shown in bold.")
    elseif !p.trace
        println(io, "\nMetrics are averaged over all launches of each kernel.")
    end

    return
end

function show_metrics(io::IOContext{IOBuffer}, rows::NamedTuple, values::Matrix{Float64},
                      metrics::Vector{MetricColumn}, trace::Bool, crop::Symbol)
    launch_columns = if trace
        [:id, :grid, :block, :registers]
    elseif any(col -> col.metric == duration_metric.metric, metrics)
        [:time_ratio, :calls]
    else
        [:calls]
    end
    metric_keys = [Symbol("metric", j) for j in eachindex(metrics)]
    columns = push!(append!(copy(launch_columns), metric_keys), :name)
    data = Matrix{Any}(undef, length(rows.name), length(columns))
    for (j, column) in enumerate(launch_columns)
        set_column!(data, j, rows[column])
    end
    data[:, length(launch_columns) .+ eachindex(metrics)] = values
    data[:, end] = rows.name

    labels = Dict(:id => "ID", :grid => "Blocks", :block => "Threads",
                  :registers => "Regs", :time_ratio => "Time (%)", :calls => "Calls",
                  :name => "Name")
    for (key, col) in zip(metric_keys, metrics)
        labels[key] = col.kind == :time && !trace ? "Avg time" : col.name
    end
    alignment = [k == :name ? :l : :r for k in columns]
    metric_kind = Dict(zip(metric_keys, [col.kind for col in metrics]))

    function formatter(v, column)
        if column == :time_ratio
            format_percentage(v)
        elseif haskey(metric_kind, column)
            format_metric(v, metric_kind[column])
        elseif v isa CuDim3
            format_dim(v)
        else
            v
        end
    end

    # emphasize units that come close to their peak throughput
    highlight_peak = TextHighlighter(crayon"bold") do data, i, j
        get(metric_kind, columns[j], :none) == :percent && data[i, j]::Float64 >= 80
    end

    print_table(io, crop, data, columns; labels, alignment, formatter,
                highlighters=[highlight_peak])
end

function format_metric(v::Float64, kind::Symbol)
    if kind == :time
        format_time(v / 1e9)
    elseif kind == :percent
        @sprintf("%.1f%%", v)
    elseif kind == :bytes
        Base.format_bytes(round(Int, v))
    else
        string(round(v; sigdigits=4))
    end
end

function demangle_names!(names::Vector{String})
    isempty(names) && return names
    demangled = split(readchomp(pipeline(IOBuffer(join(names, '\n')), `$(demumble())`)), '\n')
    copy!(names, demangled)
end


#
# JSON output
#

# the reports contain few kinds of values, so we don't need a JSON library to write them.
# objects and arrays are built from untyped containers, so that writing a report doesn't
# need to compile code for every combination of values (e.g. missing or not) it contains.
# missing values and non-finite numbers (e.g. the duration of an unterminated NVTX range)
# are written as `null`.
struct JSONObject
    pairs::Vector{Pair{String,Any}}
end
JSONObject(@nospecialize(pairs::Pair...)) = JSONObject(Pair{String,Any}[pairs...])
json(io::IO, ::Union{Nothing,Missing}) = print(io, "null")
json(io::IO, x::Union{Bool,Integer}) = print(io, x)
json(io::IO, x::AbstractFloat) = isfinite(x) ? print(io, Float64(x)) : print(io, "null")
json(io::IO, x::CuDim3) = print(io, '[', Int(x.x), ',', Int(x.y), ',', Int(x.z), ']')
function json(io::IO, x::String)
    print(io, '"')
    for c in x
        if c == '"' || c == '\\'
            print(io, '\\', c)
        elseif c < ' '
            print(io, "\\u", string(UInt32(c); base=16, pad=4))
        else
            print(io, c)
        end
    end
    print(io, '"')
end
function json(io::IO, x::Vector{Any})
    print(io, '[')
    for (i, v) in enumerate(x)
        i > 1 && print(io, ',')
        json(io, v)
    end
    print(io, ']')
end
function json(io::IO, x::JSONObject)
    print(io, '{')
    for (i, (k, v)) in enumerate(x.pairs)
        i > 1 && print(io, ',')
        json(io, k)
        print(io, ':')
        json(io, v)
    end
    print(io, '}')
end

function Base.show(io::IO, ::MIME"application/json", report::Union{ProfileResults,KernelProfile})
    # see `show_report`
    buf = IOBuffer()
    json(buf, json_report(report))
    write(io, take!(buf))
    return
end

# times in reports are in seconds, with rounding errors from converting nanoseconds
ns(t) = t === missing ? missing : round(t * 1e9; digits=1)
fraction(x) = round(x; digits=6)

json_report(results::ProfileResults) =
    barrier(json_report, report(results), results.trace, results.raw)
function json_report(r::NamedTuple, trace::Bool, raw::Bool)

    summary(df) = Any[let dist = df.time_dist[i]
        JSONObject("name" => df.name[i], "calls" => df.calls[i],
                   "total_ns" => ns(df.time[i]), "time_fraction" => fraction(df.time_ratio[i]),
                   "mean_ns" => ns(dist === missing ? df.time[i] : dist.mean),
                   "std_ns" => (dist === missing ? nothing : ns(dist.std)),
                   "min_ns" => ns(dist === missing ? df.time[i] : dist.min),
                   "max_ns" => ns(dist === missing ? df.time[i] : dist.max))
    end for i in eachindex(df.name)]

    host, device, nvtx = if trace
        df = r.host
        host = Any[JSONObject("id" => df.id[i], "start_ns" => ns(df.start[i]),
                              "duration_ns" => ns(df.time[i]), "thread" => df.tid[i],
                              "name" => df.name[i], "details" => df.details[i])
                   for i in eachindex(df.name)]
        df = r.device
        device = Any[let shared_mem = df.shared_mem[i], local_mem = df.local_mem[i]
            JSONObject("id" => df.id[i], "start_ns" => ns(df.start[i]),
                       "duration_ns" => ns(df.time[i]), "device" => df.device[i],
                       "context" => df.context[i], "stream" => df.stream[i],
                       "name" => df.name[i], "grid" => df.grid[i], "block" => df.block[i],
                       "registers" => df.registers[i],
                       "static_shared_mem_bytes" => (shared_mem === missing ? missing : shared_mem.static),
                       "dynamic_shared_mem_bytes" => (shared_mem === missing ? missing : shared_mem.dynamic),
                       "local_mem_bytes_per_thread" => (local_mem === missing ? missing : local_mem.thread),
                       "local_mem_bytes" => (local_mem === missing ? missing : local_mem.total),
                       "size_bytes" => df.size[i],
                       "throughput_bytes_per_second" => df.throughput[i],
                       "details" => df.details[i])
        end for i in eachindex(df.name)]
        df = r.nvtx
        nvtx = Any[JSONObject("id" => df.id[i], "start_ns" => ns(df.start[i]),
                              "duration_ns" => ns(df.time[i]), "thread" => df.tid[i],
                              "domain" => df.domain[i], "name" => df.name[i],
                              "payload" => df.payload[i])
                   for i in eachindex(df.name)]
        host, device, nvtx
    else
        summary(r.host), summary(r.device), summary(r.nvtx)
    end

    JSONObject("schema_version" => 1, "view" => (trace ? "trace" : "summary"),
               "raw" => raw, "duration_ns" => ns(r.time), "events" => r.events,
               "host_time_ns" => ns(r.host_time), "device_time_ns" => ns(r.device_time),
               "host" => host, "device" => device, "nvtx" => nvtx)
end

const metric_units = Dict(:time => "nanosecond", :percent => "percent", :bytes => "byte")

json_report(p::KernelProfile) = barrier(json_report, p, report(p))
function json_report(p::KernelProfile, r::NamedTuple)
    (; rows, values) = r
    metrics(i) = JSONObject(Pair{String,Any}[metric => values[i, j]
                                             for (j, metric) in enumerate(p.metrics)])

    kernels = if p.trace
        Any[JSONObject("id" => rows.id[i], "name" => rows.name[i], "grid" => rows.grid[i],
                       "block" => rows.block[i], "registers" => rows.registers[i],
                       "stream" => rows.stream[i], "metrics" => metrics(i))
            for i in eachindex(rows.name)]
    else
        Any[JSONObject("name" => rows.name[i], "calls" => rows.calls[i],
                       "time_fraction" => fraction(rows.time_ratio[i]),
                       "metrics" => metrics(i))
            for i in eachindex(rows.name)]
    end

    JSONObject("schema_version" => 1, "view" => (p.trace ? "trace" : "summary"),
               "device" => p.device, "launches" => length(p.kernels.id), "passes" => p.passes,
               "metrics" => Any[JSONObject("name" => col.metric, "label" => col.name,
                                           "unit" => get(metric_units, col.kind, nothing))
                                for col in p.columns],
               "kernels" => kernels)
end

#
# benchmarking
#

function benchmark_and_profile(f; time=1.0, kwargs...)
    # warm-up
    f()

    # the benchmarking code; pretty naive right now
    function benchmark_harness(; kwargs...)
        t0 = time_ns()
        domain = NVTX.Domain("@bprofile")
        while (time_ns() - t0)/1e9 < time
            NVTX.@range "iteration" domain=domain begin
                f()
                CUDACore.cuCtxSynchronize()
            end
        end
    end

    profile_internally(benchmark_harness; kwargs...)
end

end
