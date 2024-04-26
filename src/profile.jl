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
may not be relevant. The output will be written to `io`, which defaults to `stdout`.

Slow operations will be highlighted in the output: Entries colored in yellow are among the
slowest 25%, while entries colored in red are among the slowest 5% of all operations.

!!! compat "Julia 1.9" This functionality is only available on Julia 1.9 and later.

!!! compat "CUDA 11.2" Older versions of CUDA, before 11.2, contain bugs that may prevent
    the `CUDA.@profile` macro to work. It is recommended to use a newer runtime.

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


module Profile

using ..CUDA
using ..NVTX
using ..CUPTI

using PrettyTables
using DataFrames
using Statistics
using Crayons
using Printf


#
# external profiler
#

function profile_externally(f)
    # wait for the device to become idle (and trigger a GC to avoid interference)
    CUDA.cuCtxSynchronize()
    GC.gc(false)
    GC.gc(true)

    start()
    try
        f()
    finally
        stop()
    end
end

const _cupti_active = Ref{Union{Nothing,Bool}}(nothing)
function detect_cupti()
    if _cupti_active[] !== nothing
        return _cupti_active[]
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
        nsight = nothing
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

const __nsight = Ref{Union{Nothing,String}}()
function nsight()
    if !isassigned(__nsight)
        # find the active Nsight Systems profiler
        if haskey(ENV, "NSYS_PROFILING_SESSION_ID") && ccall(:jl_generating_output, Cint, ()) == 0
            __nsight[] = find_nsys()
            @assert isfile(__nsight[])
            @info "Running under Nsight Systems, CUDA.@profile will automatically start the profiler"
        else
            __nsight[] = nothing
        end
    end

    __nsight[]
end


"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
function start()
    if nsight() !== nothing
        run(`$(nsight()) start --capture-range=cudaProfilerApi`)
        # it takes a while for the profiler to actually start tracing our process
        sleep(0.01)
    end
    CUDA.cuProfilerStart()
end

"""
    stop()

Disables profile collection by the active profiling tool for the current context. If
profiling is already disabled, then this call has no effect.
"""
function stop()
    CUDA.cuProfilerStop()
    if nsight() !== nothing
        @info """Profiling has finished, open the report listed above with `nsys-ui`
                 If no report was generated, try launching `nsys` with `--trace=cuda`"""
    end
end


#
# integrated profiler
#

"""
    ProfileResults(...)

The results of a profiling run, as returned by [`@profile`](@ref). The recommended way to
interpret these results is to visualize them using the I/O stack (e.g. by calling `display`,
`print`, `string`, ...)

For programmatic access, it is possible to access the fields of this struct. However, the
exact format is not guaranteed to be stable, and may change between CUDA.jl releases.
Currently, it contains three dataframes:
- `host`, containing host-side activity;
- `device`, containing device-side activity;
- `nvtx`, with information on captured NVTX ranges and events.

See also: [`@profile`](@ref)
"""
Base.@kwdef struct ProfileResults
    # captured data
    host::DataFrame
    device::DataFrame
    nvtx::DataFrame

    # display properties set by `@profile` kwargs
    trace::Bool=false
    raw::Bool=false
end

function profile_internally(f; concurrent=true, kwargs...)
    activity_kinds = [
        # API calls
        CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
        CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME,
        # kernel execution
        concurrent ? CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL : CUPTI.CUPTI_ACTIVITY_KIND_KERNEL,
        CUPTI.CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API,
        # memory operations
        CUPTI.CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI.CUPTI_ACTIVITY_KIND_MEMSET,
        # NVTX markers
        CUPTI.CUPTI_ACTIVITY_KIND_MARKER,
    ]
    if CUDA.runtime_version() >= v"11.2"
        # additional information for API host calls
        push!(activity_kinds, CUPTI.CUPTI_ACTIVITY_KIND_MEMORY2)
    else
        @warn "The integrated profiler is not supported on CUDA <11.2, and may fail." maxlog=1
    end
    if CUDA.runtime_version() >= v"12.0"
        # additional data on NVTX markers
        push!(activity_kinds, CUPTI.CUPTI_ACTIVITY_KIND_MARKER_DATA)
    end
    if VERSION < v"1.9"
        @error "The integrated profiler is not supported on Julia <1.9, and will crash." maxlog=1
    end
    cfg = CUPTI.ActivityConfig(activity_kinds)

    # wait for the device to become idle (and trigger a GC to avoid interference)
    CUDA.cuCtxSynchronize()

    CUPTI.enable!(cfg) do
        # sink the initial profiler overhead into a synchronization call
        CUDA.cuCtxSynchronize()

        f()

        # synchronize to ensure we capture all activity
        CUDA.cuCtxSynchronize()
    end

    data = capture(cfg)
    ProfileResults(; data..., kwargs...)
end

# convert CUPTI activity records to host and device traces
function capture(cfg)
    host_trace = DataFrame(
        id      = Int[],
        start   = Float64[],
        stop    = Float64[],
        name    = String[],

        tid    = Int[],
    )
    device_trace = DataFrame(
        id      = Int[],
        start   = Float64[],
        stop    = Float64[],
        name    = String[],

        device  = Int[],
        context = Int[],
        stream  = Int[],

        # kernel launches
        grid            = Union{Missing,CUDA.CuDim3}[],
        block           = Union{Missing,CUDA.CuDim3}[],
        registers       = Union{Missing,Int64}[],
        shared_mem      = Union{Missing,@NamedTuple{static::Int64,dynamic::Int64}}[],
        local_mem       = Union{Missing,@NamedTuple{thread::Int64,total::Int64}}[],

        # memory operations
        size        = Union{Missing,Int64}[],
    )
    details = DataFrame(
        id      = Int[],
        details = String[],
    )
    nvtx_trace = DataFrame(
        id      = Int[],
        start   = Float64[],
        type    = Symbol[],
        tid     = Int[],
        name    = Union{Missing,String}[],
        domain  = Union{Missing,String}[],
    )
    nvtx_data = DataFrame(
        id       = Int[],
        payload  = Any[],
        color    = Union{Nothing,UInt32}[],
        category = UInt32[],
    )

    # memory_kind fields are sometimes typed CUpti_ActivityMemoryKind, sometimes UInt
    as_memory_kind(x) = isa(x, CUPTI.CUpti_ActivityMemoryKind) ? x : CUPTI.CUpti_ActivityMemoryKind(x)

    cuda_version = CUDA.runtime_version()
    CUPTI.process(cfg) do ctx, stream_id, record
        # driver API calls
        if record.kind in [CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
                           CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME,
                           CUPTI.CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API]
            id = record.correlationId
            t0, t1 = record.start/1e9, record._end/1e9

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

            push!(host_trace, (; id, start=t0, stop=t1, name,
                                 tid=record.threadId))

        # memory operations
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MEMCPY
            id = record.correlationId
            t0, t1 = record.start/1e9, record._end/1e9

            src_kind = as_memory_kind(record.srcKind)
            dst_kind = as_memory_kind(record.dstKind)
            name = "[copy $(string(src_kind)) to $(string(dst_kind)) memory]"

            push!(device_trace, (; id, start=t0, stop=t1, name,
                                   device=record.deviceId,
                                   context=record.contextId,
                                   stream=record.streamId,
                                   size=record.bytes); cols=:union)
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MEMSET
            id = record.correlationId
            t0, t1 = record.start/1e9, record._end/1e9

            memory_kind = as_memory_kind(record.memoryKind)
            name = "[set $(string(memory_kind)) memory]"

            push!(device_trace, (; id, start=t0, stop=t1, name,
                                   device=record.deviceId,
                                   context=record.contextId,
                                   stream=record.streamId,
                                   size=record.bytes); cols=:union)

        # memory allocations
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MEMORY2 && cuda_version >= v"11.2"
            # XXX: we'd prefer to postpone processing (i.e. calling format_bytes),
            #      but cannot realistically add a column for every API call

            id = record.correlationId

            memory_kind = as_memory_kind(record.memoryKind)
            str = "$(Base.format_bytes(record.bytes)), $(string(memory_kind)) memory"

            push!(details, (id, str))

        # kernel execution
        # TODO: CUPTI_ACTIVITY_KIND_CDP_KERNEL (CUpti_ActivityCdpKernel)
        elseif record.kind in [CUPTI.CUPTI_ACTIVITY_KIND_KERNEL,
                               CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL]
            id = record.correlationId
            t0, t1 = record.start/1e9, record._end/1e9

            name = unsafe_string(record.name)
            grid = CUDA.CuDim3(record.gridX, record.gridY, record.gridZ)
            block = CUDA.CuDim3(record.blockX, record.blockY, record.blockZ)
            registers = record.registersPerThread
            shared_mem = (static=Int64(record.staticSharedMemory),
                          dynamic=Int64(record.dynamicSharedMemory))
            local_mem = (thread=Int64(record.localMemoryPerThread),
                         total=Int64(record.localMemoryTotal))

            push!(device_trace, (; id, start=t0, stop=t1, name,
                                   device=record.deviceId,
                                   context=record.contextId,
                                   stream=record.streamId,
                                   grid, block, registers,
                                   shared_mem, local_mem); cols=:union)

        # NVTX markers
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MARKER
            start = record.timestamp/1e9
            name = record.name == C_NULL ? missing : unsafe_string(record.name)
            domain = record.domain == C_NULL ? missing : unsafe_string(record.domain)

            if record.flags == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS
                @assert record.objectKind == CUDA.CUPTI.CUPTI_ACTIVITY_OBJECT_THREAD
                tid = record.objectId.pt.threadId
                push!(nvtx_trace, (; record.id, start, tid, type=:instant, name, domain))
            elseif record.flags == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_START
                @assert record.objectKind == CUDA.CUPTI.CUPTI_ACTIVITY_OBJECT_THREAD
                tid = record.objectId.pt.threadId
                push!(nvtx_trace, (; record.id, start, tid, type=:start, name, domain))
            elseif record.flags == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_END
                @assert record.objectKind == CUDA.CUPTI.CUPTI_ACTIVITY_OBJECT_THREAD
                tid = record.objectId.pt.threadId
                push!(nvtx_trace, (; record.id, start, tid, type=:end, name, domain))
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

            color = if record.flags & CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE
                nothing
            elseif record.flags & CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB == CUPTI.CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB
                record.color
            else
                @error "Unexpected CUPTI marker color flag $(Int(record.flags)). Please file an issue."
                nothing
            end

            push!(nvtx_data, (; record.id, payload, color, record.category))
        else
            @error "Unexpected CUPTI activity kind $(Int(record.kind)). Please file an issue."
        end
    end

    # merge in the details
    host_trace = leftjoin(host_trace, details, on=:id, order=:left)
    device_trace = leftjoin(device_trace, details, on=:id, order=:left)
    nvtx_trace = leftjoin(nvtx_trace, nvtx_data, on=:id, order=:left)

    return (; host=host_trace, device=device_trace, nvtx=nvtx_trace)
end

function Base.show(io::IO, results::ProfileResults)
    results = deepcopy(results)

    # find the relevant part of the trace (marked by calls to 'cuCtxSynchronize')
    trace_first_sync = findfirst(results.host.name .== "cuCtxSynchronize")
    trace_first_sync === nothing && error("Could not find the start of the profiling data.")
    trace_last_sync = findlast(results.host.name .== "cuCtxSynchronize")
    trace_first_sync == trace_last_sync && error("Could not find the end of the profiling data.")
    ## truncate the trace
    if !results.raw || !results.trace
        trace_begin = results.host.stop[trace_first_sync]
        trace_end = results.host.stop[trace_last_sync]

        trace_first_call = copy(results.host[trace_first_sync+1, :])
        trace_last_call = copy(results.host[trace_last_sync-1, :])
        for df in (results.host, results.device)
            filter!(row -> trace_first_call.id <= row.id <= trace_last_call.id, df)
        end
        trace_divisions = Int[]
    else
        # in raw mode, we display the entire trace, but highlight the relevant part.
        # note that we only do so when tracing, because otherwise the summary would
        # be skewed by the expensive initial API call used to sink the profiler overhead.
        trace_divisions = [trace_first_sync, trace_last_sync-1]

        # inclusive bounds
        trace_begin = results.host.start[begin]
        trace_end = results.host.stop[end]
    end
    trace_time = trace_end - trace_begin

    # compute event and trace duration
    for df in (results.host, results.device)
        df.time = df.stop .- df.start
    end
    events = nrow(results.host) + nrow(results.device)
    println(io, "Profiler ran for $(format_time(trace_time)), capturing $(events) events.")

    # make some numbers more friendly to read
    ## make timestamps relative to the start
    for df in (results.host, results.device)
        df.start .-= trace_begin
        df.stop .-= trace_begin
    end
    results.nvtx.start .-= trace_begin
    if !results.raw
        # renumber event IDs from 1
        first_id = minimum([results.host.id; results.device.id])
        for df in (results.host, results.device)
            df.id .-= first_id - 1
        end

        # renumber thread IDs from 1
        threads = unique([results.host.tid; results.nvtx.tid])
        for df in (results.host, results.nvtx)
            broadcast!(df.tid, df.tid) do tid
                findfirst(isequal(tid), threads)
            end
        end

    end

    # helper function to visualize slow trace entries
    function time_highlighters(df)
        ## filter out entries that execute _very_ quickly (like calls to cuCtxGetCurrent)
        relevant_times = df[df.time .>= 1e-8, :time]

        isempty(relevant_times) && return ()
        p75 = quantile(relevant_times, 0.75)
        p95 = quantile(relevant_times, 0.95)

        highlight_p95 = Highlighter((data, i, j) -> (names(data)[j] == "time") &&
                                                    (data[i,j] >= p95),
                                    crayon"red")
        highlight_p75 = Highlighter((data, i, j) -> (names(data)[j] == "time") &&
                                                    (data[i,j] >= p75),
                                    crayon"yellow")
        highlight_bold = Highlighter((data, i, j) -> (names(data)[j] == "name") &&
                                                     (data[!, :time][i] >= p75),
                                    crayon"bold")

        (highlight_p95, highlight_p75, highlight_bold)
    end

    function summarize_trace(df)
        df = groupby(df, :name)

        # gather summary statistics
        function analyze_time(time)
            if length(time) == 1
                missing
            else
                Ref((; std=std(time), mean=mean(time), min=minimum(time), max=maximum(time)))
            end
        end
        df = combine(df,
            :time => sum           => :time,
            :time => length        => :calls,
            :time => analyze_time  => :time_dist,
        )
        df.time_ratio = df.time ./ trace_time
        sort!(df, :time_ratio, rev=true)

        return df
    end

    trace_column_names = Dict(
        "id"            => "ID",
        "start"         => "Start",
        "time"          => "Time",
        "grid"          => "Blocks",
        "tid"           => "Thread",
        "block"         => "Threads",
        "registers"     => "Regs",
        "shared_mem"    => "Shared Mem",
        "local_mem"     => "Local Mem",
        "size"          => "Size",
        "throughput"    => "Throughput",
        "device"        => "Device",
        "stream"        => "Stream",
        "name"          => "Name",
        "domain"        => "Domain",
        "details"       => "Details",
        "payload"       => "Payload"
    )

    summary_column_names = Dict(
        "time"          => "Total time",
        "time_ratio"    => "Time (%)",
        "calls"         => "Calls",
        "time_dist"     => "Time distribution",
        "name"          => "Name"
    )

    summary_formatter(df) = function(v, i, j)
        if names(df)[j] == "time_ratio"
            format_percentage(v)
        elseif names(df)[j] == "time"
            format_time(v)
        elseif names(df)[j] == "time_dist"
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

    # host-side activity
    let
        # to determine the time the host was active, we should look at threads separately
        host_time = maximum(combine(groupby(results.host, :tid), :time => sum => :time).time)
        host_ratio = host_time / trace_time

        # get rid of API call version suffixes
        results.host.name = replace.(results.host.name, r"_v\d+$" => "")

        df = if results.raw
            results.host
        else
            # filter spammy API calls
            filter(results.host) do row
                !in(row.name, [# context and stream queries we use for nonblocking sync
                               "cuCtxGetCurrent", "cuCtxGetId", "cuCtxGetApiVersion",
                               "cuStreamQuery", "cuStreamGetId",
                               # occupancy API, done before every kernel launch
                               "cuOccupancyMaxPotentialBlockSize",
                               # driver pointer set-up
                               "cuGetProcAddress",
                               # called a lot during compilation
                               "cuDeviceGetAttribute",
                               # done before every memory operation
                               "cuPointerGetAttribute", "cuDeviceGetMemPool"])
            end
        end

        # instantaneous NVTX markers can be added to the API trace
        if results.trace
            markers = copy(results.nvtx[results.nvtx.type .== :instant, :])
            markers.id .= missing
            markers.time .= 0.0
            markers.details = map(markers.name, markers.domain) do name, domain
                if name !== missing && domain !== missing
                    "$(domain).$(name)"
                elseif name !== missing
                    "$name"
                end
            end
            markers.name .= "NVTX marker"
            append!(df, markers; cols=:subset)
            sort!(df, :start)
        end

        if !isempty(df)
            println(io, "\nHost-side activity: calling CUDA APIs took $(format_time(host_time)) ($(format_percentage(host_ratio)) of the trace)")
        end
        if isempty(df)
            println(io, "\nNo host-side activity was recorded.")
        elseif results.trace
            # determine columns to show, based on whether they contain useful information
            columns = [:id, :start, :time]
            for col in [:tid]
                if results.raw || length(unique(df[!, col])) > 1
                    push!(columns, col)
                end
            end
            push!(columns, :name)
            if any(!ismissing, df.details)
                push!(columns, :details)
            end

            df = df[:, columns]

            header = [trace_column_names[name] for name in names(df)]
            alignment = [name in ["name"] ? :l : :r for name in names(df)]
            formatters = function(v, i, j)
                if v === missing
                    return "-"
                elseif names(df)[j] in ["start", "time"]
                    format_time(v)
                else
                    v
                end
            end
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters, highlighters, crop,
                                 body_hlines=trace_divisions)
        else
            df = summarize_trace(df)

            columns = [:time_ratio, :time, :calls]
            if any(!ismissing, df.time_dist)
                push!(columns, :time_dist)
            end
            push!(columns, :name)
            df = df[:, columns]

            header = [summary_column_names[name] for name in names(df)]
            alignment = [name in ["name", "time_dist"] ? :l : :r for name in names(df)]
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters=summary_formatter(df), highlighters, crop)
        end
    end

    # device-side activity
    let
        device_time = sum(results.device.time)
        device_ratio = device_time / trace_time
        if !isempty(results.device)
            println(io, "\nDevice-side activity: GPU was busy for $(format_time(device_time)) ($(format_percentage(device_ratio)) of the trace)")
        end

        # add memory throughput information
        results.device.throughput = results.device.size ./ results.device.time

        if isempty(results.device)
            println(io, "\nNo device-side activity was recorded.")
        elseif results.trace
            # determine columns to show, based on whether they contain useful information
            columns = [:id, :start, :time]
            ## device/stream identification
            for col in [:device, :stream]
                if results.raw || length(unique(results.device[!, col])) > 1
                    push!(columns, col)
                end
            end
            ## kernel details (can be missing)
            for col in [:block, :grid, :registers]
                if results.raw || any(!ismissing, results.device[!, col])
                    push!(columns, col)
                end
            end
            if results.raw || any(val->!ismissing(val) && (val.static > 0 || val.dynamic > 0), results.device.shared_mem)
                push!(columns, :shared_mem)
            end
            if results.raw || any(val->!ismissing(val) && val.thread > 0, results.device.local_mem)
                push!(columns, :local_mem)
            end
            ## memory details (can be missing)
            if results.raw || any(!ismissing, results.device.size)
                push!(columns, :size)
                push!(columns, :throughput)
            end
            push!(columns, :name)

            df = results.device[:, columns]

            header = [trace_column_names[name] for name in names(df)]
            alignment = [name in ["name"] ? :l : :r for name in names(df)]
            formatters = function(v, i, j)
                if v === missing
                    return "-"
                elseif names(df)[j] in ["start", "time"]
                    format_time(v)
                elseif names(df)[j] in ["size"]
                    Base.format_bytes(v)
                elseif names(df)[j] in ["shared_mem"]
                    if results.raw || v.static > 0 && v.dynamic > 0
                        "$(Base.format_bytes(v.static)) static, $(Base.format_bytes(v.dynamic)) dynamic"
                    elseif v.static > 0
                        "$(Base.format_bytes(v.static)) static"
                    elseif v.dynamic > 0
                        "$(Base.format_bytes(v.dynamic)) dynamic"
                    else
                        "-"
                    end
                elseif names(df)[j] in ["local_mem"]
                    "$(Base.format_bytes(v.thread)) / $(Base.format_bytes(v.total))"
                elseif names(df)[j] in ["throughput"]
                    Base.format_bytes(v) * "/s"
                elseif names(df)[j] in ["device"]
                    CUDA.name(CuDevice(v))
                elseif v isa CUDA.CuDim3
                    if v.z != 1
                        "$(Int(v.x))×$(Int(v.y))×$(Int(v.z))"
                    elseif v.y != 1
                        "$(Int(v.x))×$(Int(v.y))"
                    else
                        "$(Int(v.x))"
                    end
                else
                    v
                end
            end
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters, highlighters, crop,
                                 body_hlines=trace_divisions)
        else
            df = summarize_trace(results.device)

            columns = [:time_ratio, :time, :calls]
            if any(!ismissing, df.time_dist)
                push!(columns, :time_dist)
            end
            push!(columns, :name)
            df = df[:, columns]

            header = [summary_column_names[name] for name in names(df)]
            alignment = [name in ["name", "time_dist"] ? :l : :r for name in names(df)]
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters=summary_formatter(df), highlighters, crop)
        end
    end

    # show NVTX ranges
    # TODO: do we also want to repeat the host/device summary for each NVTX range?
    #       that's what nvprof used to do, but it's a little verbose...
    nvtx_ranges = copy(results.nvtx[results.nvtx.type .== :start, :])
    nvtx_ranges = leftjoin(nvtx_ranges, results.nvtx[results.nvtx.type .== :end,
                           [:id, :start]],
                           on=:id, makeunique=true)
    if !isempty(nvtx_ranges)
        println(io, "\nNVTX ranges:")

        rename!(nvtx_ranges, :start_1 => :stop)
        nvtx_ranges.time .= nvtx_ranges.stop .- nvtx_ranges.start

        df = nvtx_ranges
        if results.trace
            # determine columns to show, based on whether they contain useful information
            columns = [:id, :start, :time]
            for col in [:tid]
                if results.raw || length(unique(df[!, col])) > 1
                    push!(columns, col)
                end
            end
            for col in [:domain, :name, :payload]
                if results.raw || any(!ismissing, df[!, col])
                    push!(columns, col)
                end
            end

            # use color information as provided by NVTX
            color_highlighters = []
            for color in unique(df.color)
                if color !== nothing
                    ids = df[df.color .== color, :id]
                    highlighter = Highlighter(Crayon(; foreground=color)) do data, i, j
                        names(data)[j] in ["name", "domain"] && data[!, :id][i] in ids
                    end
                    push!(color_highlighters, highlighter)
                end
            end

            df = df[:, columns]

            header = [trace_column_names[name] for name in names(df)]
            alignment = [name in ["name"] ? :l : :r for name in names(df)]
            formatters = function(v, i, j)
                if v === missing
                    return "-"
                elseif names(df)[j] in ["start", "time"]
                    format_time(v)
                else
                    v
                end
            end
            highlighters = tuple(color_highlighters..., time_highlighters(df)...)
            pretty_table(io, df; header, alignment, formatters, highlighters, crop)
        else
            # merge the domain and name into a single column
            nvtx_ranges.name = map(nvtx_ranges.name, nvtx_ranges.domain) do name, domain
                if name !== missing && domain !== missing
                    "$(domain).$(name)"
                elseif name !== missing
                    "$name"
                end
            end

            df = summarize_trace(nvtx_ranges)

            columns = [:time_ratio, :time, :calls]
            if any(!ismissing, df.time_dist)
                push!(columns, :time_dist)
            end
            push!(columns, :name)
            df = df[:, columns]

            header = [summary_column_names[name] for name in names(df)]
            alignment = [name in ["name", "time_dist"] ? :l : :r for name in names(df)]
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters=summary_formatter(df), highlighters, crop)
        end
    end

    return
end

format_percentage(x::Number) = @sprintf("%.2f%%", x * 100)

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
                CUDA.cuCtxSynchronize()
            end
        end
    end

    profile_internally(benchmark_harness; kwargs...)
end

end
