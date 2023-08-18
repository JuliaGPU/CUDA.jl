# Profiler control

"""
    @profile [io=stdout] [host=true] [device=true] [trace=false] [raw=false] code...
    @profile external=true code...

Profile the GPU execution of `code`.

There are two modes of operation, depending on whether `external` is `true` or `false`.

## Integrated profiler (`external=false`, the default)

In this mode, CUDA.jl will profile the execution of `code` and display the result. By
default, both host-side and device-side activity is captured; this can be controlled with
the `host` and `device` keyword arguments. If `trace` is `true`, a chronological trace of
the captured activity will be generated, where the ID column can be used to match host-side
and device-side activity; by default, only a summary will be shown. If `raw` is `true`, all
data will always be included, even if it may not be relevant. The output will be written to
`io`, which defaults to `stdout`.

Slow operations will be highlighted in the output: Entries colored in yellow are among the
slowest 25%, while entries colored in red are among the slowest 5% of all operations.

!!! compat "Julia 1.9"
    This functionality is only available on Julia 1.9 and later.

!!! compat "CUDA 11.2"
    Older versions of CUDA, before 11.2, contain bugs that may prevent the
    `CUDA.@profile` macro to work. It is recommended to use a newer runtime.

## External profilers (`external=true`)

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
    external = false
    remaining_kwargs = []
    for kwarg in kwargs
        if Meta.isexpr(kwarg, :(=))
            key, value = kwarg.args
            if key == :external
                isa(value, Bool) || throw(ArgumentError("Invalid value for keyword argument `external`: got `$value`, expected literal boolean value"))
                external = value
            else
                push!(remaining_kwargs, kwarg)
            end
        else
            throw(ArgumentError("Invalid keyword argument to CUDA.@profile: $kwarg"))
        end
    end

    if external
        Profile.emit_external_profile(code, remaining_kwargs)
    else
        Profile.emit_integrated_profile(code, remaining_kwargs)
    end
end


module Profile

using ..CUDA

using ..CUPTI

using PrettyTables
using DataFrames
using Statistics
using Crayons
using Printf


#
# external profiler
#

function emit_external_profile(code, kwargs)
    isempty(kwargs) || throw(ArgumentError("External profiler does not support keyword arguments"))

    quote
        # wait for the device to become idle (and trigger a GC to avoid interference)
        CUDA.cuCtxSynchronize()
        GC.gc(false)
        GC.gc(true)

        $Profile.start()
        try
            $(esc(code))
        finally
            $Profile.stop()
        end
    end
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

function emit_integrated_profile(code, kwargs)
    activity_kinds = [
        # API calls
        CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
        CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME,
        # kernel execution
        CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI.CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API,
        # memory operations
        CUPTI.CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI.CUPTI_ACTIVITY_KIND_MEMSET,
        # NVTX
        CUPTI.CUPTI_ACTIVITY_KIND_NAME,
        CUPTI.CUPTI_ACTIVITY_KIND_MARKER,
    ]
    if CUDA.runtime_version() >= v"11.2"
        # additional information for API host calls
        push!(activity_kinds, CUPTI.CUPTI_ACTIVITY_KIND_MEMORY2)
    else
        @warn "The integrated profiler is not supported on CUDA <11.2, and may fail." maxlog=1
    end
    if VERSION < v"1.9"
        @error "The integrated profiler is not supported on Julia <1.9, and will crash." maxlog=1
    end

    quote
        cfg = CUPTI.ActivityConfig($activity_kinds)

        # wait for the device to become idle (and trigger a GC to avoid interference)
        CUDA.cuCtxSynchronize()
        GC.gc(false)
        GC.gc(true)

        CUPTI.enable!(cfg)

        # sink the initial profiler overhead into a synchronization call
        CUDA.cuCtxSynchronize()
        try
            rv = $(esc(code))

            # synchronize to ensure we capture all activity
            CUDA.cuCtxSynchronize()

            rv
        finally
            CUPTI.disable!(cfg)
            # convert CUPTI activity records to host and device traces
            traces = $Profile.generate_traces(cfg)
            $Profile.render_traces(traces...; $(map(esc, kwargs)...))
        end
    end
end

# convert CUPTI activity records to host and device traces
function generate_traces(cfg)
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
        static_shmem    = Union{Missing,Int64}[],
        dynamic_shmem   = Union{Missing,Int64}[],

        # memory operations
        size        = Union{Missing,Int64}[],
    )
    details = DataFrame(
        id      = Int[],
        details = String[],
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
                CUPTI.cuptiGetCallbackName(CUPTI.CUPTI_CB_DOMAIN_DRIVER_API,
                                            record.cbid, ref)
                unsafe_string(ref[])
            elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_RUNTIME
                ref = Ref{Cstring}(C_NULL)
                CUPTI.cuptiGetCallbackName(CUPTI.CUPTI_CB_DOMAIN_RUNTIME_API,
                                            record.cbid, ref)
                unsafe_string(ref[])
            else
                "<unknown>"
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
            static_shmem = record.staticSharedMemory
            dynamic_shmem = record.dynamicSharedMemory

            push!(device_trace, (; id, start=t0, stop=t1, name,
                                   device=record.deviceId,
                                   context=record.contextId,
                                   stream=record.streamId,
                                   grid, block, registers,
                                   static_shmem, dynamic_shmem); cols=:union)

        # NVTX events
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_NAME
            @show record
        elseif record.kind == CUPTI.CUPTI_ACTIVITY_KIND_MARKER
            @show record

        else
            error("Unexpected CUPTI activity kind: $(record.kind). Please file an issue.")
        end
    end

    return host_trace, device_trace, details
end

# render traces to a table
function render_traces(host_trace, device_trace, details;
                       io=stdout isa Base.TTY ? IOContext(stdout, :limit => true) : stdout,
                       host=true, device=true, trace=false, raw=false)
    # find the relevant part of the trace (marked by calls to 'cuCtxSynchronize')
    trace_first_sync = findfirst(host_trace.name .== "cuCtxSynchronize")
    trace_first_sync === nothing && error("Could not find the start of the profiling trace.")
    trace_last_sync = findlast(host_trace.name .== "cuCtxSynchronize")
    trace_first_sync == trace_last_sync && error("Could not find the end of the profiling trace.")
    ## truncate the trace
    if !raw
        trace_begin = host_trace.stop[trace_first_sync]
        trace_end = host_trace.stop[trace_last_sync]

        trace_first_call = copy(host_trace[trace_first_sync+1, :])
        trace_last_call = copy(host_trace[trace_last_sync-1, :])
        for df in (host_trace, device_trace)
            filter!(row -> trace_first_call.id <= row.id <= trace_last_call.id, df)
        end
        body_hlines = Int[]
    else
        # in raw mode, we display the entire trace, but highlight the relevant part
        body_hlines = [trace_first_sync, trace_last_sync-1]

        # inclusive bounds
        trace_begin = host_trace.start[begin]
        trace_end = host_trace.stop[end]
    end
    trace_time = trace_end - trace_begin

    # compute event and trace duration
    for df in (host_trace, device_trace)
        df.time = df.stop .- df.start
    end
    events = nrow(host_trace) + nrow(device_trace)
    println(io, "Profiler ran for $(format_time(trace_time)), capturing $(events) events.")

    # make some numbers more friendly to read
    ## make timestamps relative to the start
    for df in (host_trace, device_trace)
        df.start .-= trace_begin
        df.stop .-= trace_begin
    end
    if !raw
        # renumber event IDs from 1
        first_id = minimum([host_trace.id; device_trace.id; details.id])
        for df in (host_trace, device_trace, details)
            df.id .-= first_id - 1
        end

        # renumber thread IDs from 1
        threads = unique([host_trace.tid; nvtx_trace.tid])
        for df in (host_trace, nvtx_trace)
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
        df = combine(df,
            :time => sum => :time,
            :time => length => :calls,
            :time => mean => :time_avg,
            :time => minimum => :time_min,
            :time => maximum => :time_max
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
        "static_shmem"  => "SSMem",
        "dynamic_shmem" => "DSMem",
        "size"          => "Size",
        "throughput"    => "Throughput",
        "device"        => "Device",
        "stream"        => "Stream",
        "name"          => "Name",
        "details"       => "Details"
    )

    summary_column_names = Dict(
        "time"          => "Time",
        "time_ratio"    => "Time (%)",
        "calls"         => "Calls",
        "time_avg"      => "Avg time",
        "time_min"      => "Min time",
        "time_max"      => "Max time",
        "name"          => "Name"
    )

    summary_formatter = function(v, i, j)
        if names(df)[j] == "time_ratio"
            format_percentage(v)
        elseif names(df)[j] in ["time", "time_avg", "time_min", "time_max"]
            format_time(v)
        else
            v
        end
    end

    crop = if io isa IOBuffer
        # when emitting to a string, render all content (e.g., for the tests)
        :none
    else
        :horizontal
    end

    if host
        # to determine the time the host was active, we should look at threads separately
        host_time = maximum(combine(groupby(host_trace, :tid), :time => sum => :time).time)
        host_ratio = host_time / trace_time
        println(io, "\nHost-side activity: calling CUDA APIs took $(format_time(host_time)) ($(format_percentage(host_ratio)) of the trace)")

        # get rid of API call version suffixes
        host_trace.name = replace.(host_trace.name, r"_v\d+$" => "")

        df = if raw
            host_trace
        else
            # filter spammy API calls
            filter(host_trace) do row
                !in(row.name, [# context and stream queries we use for nonblocking sync
                               "cuCtxGetCurrent", "cuStreamQuery",
                               # occupancy API, done before every kernel launch
                               "cuOccupancyMaxPotentialBlockSize",
                               # driver pointer set-up
                               "cuGetProcAddress",
                               # called a lot during compilation
                               "cuDeviceGetAttribute",
                               # pointer attribute query, done before every memory operation
                               "cuPointerGetAttribute"])
            end
        end

        # add in details
        df = leftjoin(df, details, on=:id, order=:left)

        if isempty(df)
            println(io, "No host-side activity was recorded.")
        elseif trace
            # determine columns to show, based on whether they contain useful information
            columns = [:id, :start, :time]
            for col in [:tid]
                if raw || length(unique(df[!, col])) > 1
                    push!(columns, col)
                end
            end
            push!(columns, :name)
            if any(!ismissing, df.details)
                push!(columns, :details)
            end

            df = df[:, columns]
            header = [trace_column_names[name] for name in names(df)]

            alignment = [i == lastindex(header) ? :l : :r for i in 1:length(header)]
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
            pretty_table(io, df; header, alignment, formatters, highlighters, crop, body_hlines)
        else
            df = summarize_trace(df)

            columns = [:time_ratio, :time, :calls, :time_avg, :time_min, :time_max, :name]
            df = df[:, columns]

            header = [summary_column_names[name] for name in names(df)]
            alignment = [i == lastindex(header) ? :l : :r for i in 1:length(header)]
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters=summary_formatter, highlighters, crop)
        end
    end

    if device
        device_time = sum(device_trace.time)
        device_ratio = device_time / trace_time
        println(io, "\nDevice-side activity: GPU was busy for $(format_time(device_time)) ($(format_percentage(device_ratio)) of the trace)")

        # add memory throughput information
        device_trace.throughput = device_trace.size ./ device_trace.time

        if isempty(device_trace)
            println(io, "No device-side activity was recorded.")
        elseif trace
            # determine columns to show, based on whether they contain useful information
            columns = [:id, :start, :time]
            ## device/stream identification
            for col in [:device, :stream]
                if raw || length(unique(device_trace[!, col])) > 1
                    push!(columns, col)
                end
            end
            ## kernel details (can be missing)
            for col in [:block, :grid, :registers]
                if raw || any(!ismissing, device_trace[!, col])
                    push!(columns, col)
                end
            end
            for col in [:static_shmem, :dynamic_shmem]
                if raw || any(val->!ismissing(val) && val > 0, device_trace[!, col])
                    push!(columns, col)
                end
            end
            ## memory details (can be missing)
            if raw || any(!ismissing, device_trace.size)
                push!(columns, :size)
                push!(columns, :throughput)
            end
            push!(columns, :name)

            df = device_trace[:, columns]
            header = [trace_column_names[name] for name in names(df)]

            alignment = [i == lastindex(header) ? :l : :r for i in 1:length(header)]
            formatters = function(v, i, j)
                if v === missing
                    return "-"
                elseif names(df)[j] in ["start", "time"]
                    format_time(v)
                elseif names(df)[j] in ["static_shmem", "dynamic_shmem", "size"]
                    Base.format_bytes(v)
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
            pretty_table(io, df; header, alignment, formatters, highlighters, crop, body_hlines)
        else
            df = summarize_trace(device_trace)

            columns = [:time_ratio, :time, :calls, :time_avg, :time_min, :time_max, :name]
            df = df[:, columns]

            header = [summary_column_names[name] for name in names(df)]
            alignment = [i == lastindex(header) ? :l : :r for i in 1:length(header)]
            highlighters = time_highlighters(df)
            pretty_table(io, df; header, alignment, formatters=summary_formatter, highlighters, crop)
        end
    end
end

format_percentage(x::Number) = @sprintf("%.2f%%", x * 100)

function format_time(t::Number)
    io = IOBuffer()
    if abs(t) < 1e-6  # less than 1 microsecond
        print(io, round(t * 1e9, digits=2), " ns")
    elseif abs(t) < 1e-3  # less than 1 millisecond
        print(io, round(t * 1e6, digits=2), " µs")
    elseif abs(t) < 1  # less than 1 second
        print(io, round(t * 1e3, digits=2), " ms")
    else
        print(io, round(t, digits=2), " s")
    end
    return String(take!(io))
end

end
