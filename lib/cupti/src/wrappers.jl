@public version, library_version, can_profile
@public CallbackConfig, ActivityConfig, enable!, @enable!, process

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

"""
    CUPTI.version() -> VersionNumber

The version of CUPTI's API, e.g. `v"12.9.1"` or `v"13.5.0"`, to check for API features.
Wraps `cuptiGetVersion`. Before CUDA 13, CUPTI reports a revision number, which this
function maps to the CUDA release that introduced it; several releases can share a
revision (e.g. `v"10.0"` covers CUDA 10.0 to 10.2). From CUDA 13, CUPTI reports a version
whose patch number is CUPTI's own. Either way, the version only changes when the API does,
so it can't identify features or fixes added in an update release: look for the symbol of
a new routine instead (as [`CUPTI.has_range_profiler`](@ref) does), or use
[`CUPTI.library_version`](@ref).
"""
function version()
    version_ref = Ref{Cuint}()
    cuptiGetVersion(version_ref)
    # CUPTI changed the encoding of cuptiGetVersion in CUDA 13: pre-13 it returns a small
    # index into a hand-maintained table, from 13 onwards it encodes major.minor.patch
    # directly (so >= 130000).
    if version_ref[] < 1000
        cupti_versions[version_ref[]]
    else
        major, ver = divrem(version_ref[], 10000)
        minor, patch = divrem(ver, 100)
        VersionNumber(major, minor, patch)
    end
end

"""
    CUPTI.library_version() -> VersionNumber

The release version of the loaded CUPTI library, e.g. `v"2026.4.0"`, which recent CUPTI
releases number by year. It changes with every release, so use it to work around bugs in a
specific CUPTI library, and [`CUPTI.version`](@ref) to check for API features. It is read
from the file name of the library, as NVIDIA suggests.
"""
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

"""
    can_profile() -> Bool

Whether CUDA.jl should attempt to use CUPTI in the current process.

On Tegra, CUDA 10 through 12 require root, while CUDA 13 and later require read/write
access to a profiling device node. This check is intentionally conservative because some
CUPTI versions crash in `cuptiSubscribe` when that access is missing. On other systems,
CUPTI reports permission errors normally, so this function returns `true` and lets CUPTI
handle them.
"""
function can_profile()
    CUDACore.is_tegra() || return true

    # The legacy Tegra driver does not support non-root CUPTI profiling. This was verified
    # with CUDA 10.2 and 11.4; conservatively apply the same restriction to CUDA 12.
    if CUDACore.runtime_version() < v"13"
        return Libc.geteuid() == 0
    end

    # CUDA 13 and later use per-GPU profiler nodes under /dev/nvgpu. Keep checking the
    # legacy node as well because the exact device layout is a property of the driver.
    if isdir("/dev/nvgpu")
        for dev in readdir("/dev/nvgpu"; join=true)
            _has_profiling_access(joinpath(dev, "prof")) && return true
        end
    end
    return _has_profiling_access("/dev/nvhost-prof-gpu")
end

_has_profiling_access(path) =
    ccall(:access, Cint, (Cstring, Cint), path, #=R_OK | W_OK=# 0x6) == 0


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
    CUPTI.CallbackConfig(callback, domains::Vector{CUpti_CallbackDomain}) -> CallbackConfig

Configuration for CUPTI's callback API, which calls `callback(domain, id, data)` on every
event in the callback `domains` (e.g. `CUPTI_CB_DOMAIN_DRIVER_API` or
`CUPTI_CB_DOMAIN_RESOURCE`) while the configuration is enabled with
[`CUPTI.enable!`](@ref). `id` identifies the event (e.g. a `CUpti_driver_api_trace_cbid`
for the driver API domain), and `data` is a copy of the record CUPTI passes, whose type
depends on the domain: `CUpti_CallbackData` for the driver and runtime API domains,
`CUpti_ResourceData`, `CUpti_SynchronizeData` or `CUpti_NvtxData` for the others. The
copy is shallow: memory that the record points to, such as the arguments of an API call,
is only valid until `callback` returns, so copy what you want to keep.

`callback` runs on the thread that triggered the event, possibly on several threads at
once and while CUPTI or the driver hold internal locks. It must not yield, e.g. by doing
I/O or waiting on a lock, and must not call the CUDA driver or runtime APIs, except for
what CUPTI's documentation allows.

```julia
cfg = CUPTI.CallbackConfig([CUPTI.CUPTI_CB_DOMAIN_DRIVER_API]) do domain, id, data
    # inspect data
end

CUPTI.enable!(cfg) do
    # do stuff
end
```
"""
mutable struct CallbackConfig
    callback::Function
    callback_kinds::Vector{CUpti_CallbackDomain}
end



#
# activity API
#

# memory that owns a buffer allocated by the `request` callback below. on Julia 1.11+, it
# is a `Memory`: wrapping a pointer as an `Array` there allocates the array after the
# memory took ownership, so an allocation failure in between would free the buffer twice.
const BufferMemory = VERSION >= v"1.11" ? Memory{UInt8} : Vector{UInt8}

# a buffer of activity records, returned by CUPTI
struct ActivityBuffer
    ctx::CUcontext
    stream_id::UInt32
    data::BufferMemory
    valid_size::Int
end

"""
    CUPTI.ActivityConfig(activity_kinds::Vector{CUpti_ActivityKind}) -> ActivityConfig

Configuration for CUPTI's activity API, which records the given kinds of activity (e.g.
`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` or `CUPTI_ACTIVITY_KIND_MEMCPY`) while the
configuration is enabled with [`CUPTI.enable!`](@ref), using `cuptiActivityEnable`. The
records are kept in the configuration, and can be read afterwards with
[`CUPTI.process`](@ref). Activity kinds that the loaded CUPTI doesn't support are skipped.

```julia
cfg = CUPTI.ActivityConfig([CUPTI.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL])

CUPTI.enable!(cfg) do
    # do stuff
end

CUPTI.process(cfg) do ctx, stream_id, record
    # inspect record
end
```
"""
struct ActivityConfig
    activity_kinds::Vector{CUpti_ActivityKind}

    # completed activity buffers, in the order CUPTI completed them
    results::Vector{ActivityBuffer}
end

ActivityConfig(activity_kinds) = ActivityConfig(activity_kinds, ActivityBuffer[])

const activity_lock = ReentrantLock()

# CUPTI requests activity buffers through a callback on the thread that records an
# activity, and returns them through another on a worker thread. The recording thread can
# be one that is collecting garbage, e.g. when a GC hook emits an NVTX range, where Julia
# code can't run: entering a `@cfunction` waits for the collection to finish. Both
# callbacks are therefore native functions, compiled from the LLVM IR below when they are
# first needed. `request` allocates buffers with `malloc`, and `complete` pushes them onto
# a lock-free list, which Julia takes after flushing.
#
# On Windows, the functions have no registered unwind information. Julia's stack walker
# stops at them (e.g. when the profiler samples a thread in `malloc` called by `request`),
# and nothing throws through them.
#
# They are compiled with an LLJIT rather than `llvmcall`, whose functions have no usable
# address in package images built for several CPU targets.

const buffer_size = 8 * 1024 * 1024    # CUPTI suggests 1 to 10 MB

# every buffer ends with a trailer that `complete` fills in: the next buffer in the list,
# the context, the stream ID, and the number of bytes of records
const trailer_size = 32

const callback_ir = """
    @malloc = global ptr null, align 8      ; Julia's `malloc`, so that it can free buffers
    @completed = global ptr null, align 8   ; the list of completed buffers
    @failures = global i64 0, align 8       ; the number of failed allocations

    define void @request(ptr %buffer, ptr %size, ptr %max_records) nounwind {
    entry:
        %malloc = load ptr, ptr @malloc, align 8
        %buf = call ptr %malloc(i64 $(buffer_size + trailer_size))
        %failed = icmp eq ptr %buf, null
        br i1 %failed, label %fail, label %done
    fail:
        %old = atomicrmw add ptr @failures, i64 1 monotonic, align 8
        br label %done
    done:
        %sz = select i1 %failed, i64 0, i64 $buffer_size
        store ptr %buf, ptr %buffer, align 8
        store i64 %sz, ptr %size, align 8
        store i64 0, ptr %max_records, align 8
        ret void
    }

    define void @complete(ptr %ctx, i32 %stream_id, ptr %buf, i64 %size,
                          i64 %valid_size) nounwind {
    entry:
        %null = icmp eq ptr %buf, null
        br i1 %null, label %done, label %fill
    fill:
        %trailer = getelementptr i8, ptr %buf, i64 $buffer_size
        %ctx_ptr = getelementptr i8, ptr %trailer, i64 8
        store ptr %ctx, ptr %ctx_ptr, align 8
        %stream_ptr = getelementptr i8, ptr %trailer, i64 16
        %stream = zext i32 %stream_id to i64
        store i64 %stream, ptr %stream_ptr, align 8
        %valid_ptr = getelementptr i8, ptr %trailer, i64 24
        store i64 %valid_size, ptr %valid_ptr, align 8
        %head = load atomic ptr, ptr @completed monotonic, align 8
        br label %push
    push:
        %next = phi ptr [ %head, %fill ], [ %seen, %push ]
        store ptr %next, ptr %trailer, align 8
        %pair = cmpxchg ptr @completed, ptr %next, ptr %buf release monotonic, align 8
        %seen = extractvalue { ptr, i1 } %pair, 0
        %ok = extractvalue { ptr, i1 } %pair, 1
        br i1 %ok, label %done, label %push
    done:
        ret void
    }
    """

const NativeCallbacks = @NamedTuple{request::Ptr{Cvoid}, complete::Ptr{Cvoid},
                                    completed::Ptr{Ptr{UInt8}}, failures::Ptr{Int}}

# compiled once per process, while holding `activity_lock`. `__init__` resets these, as
# compiled code doesn't survive precompilation.
const native_callbacks = Ref{Union{Nothing,NativeCallbacks}}(nothing)
const native_jit = Ref{Union{Nothing,LLVM.LLJIT}}(nothing)  # never disposed

function get_native_callbacks()
    callbacks = native_callbacks[]
    callbacks === nothing || return callbacks

    jit = LLVM.LLJIT(; tm=LLVM.JITTargetMachine())
    LLVM.@dispose ts_ctx=LLVM.ThreadSafeContext(; opaque_pointers=true) begin
        ts_mod = LLVM.ThreadSafeModule("cupti_callbacks")
        ts_mod() do mod
            LLVM.link!(mod, parse(LLVM.Module, callback_ir))
        end
        LLVM.add!(jit, jit.main_dylib, ts_mod)
    end
    address(name) = Ptr{Cvoid}(pointer(LLVM.lookup(jit, name)))
    unsafe_store!(Ptr{Ptr{Cvoid}}(address("malloc")), cglobal(:malloc))

    native_jit[] = jit
    native_callbacks[] = (; request=address("request"), complete=address("complete"),
                            completed=Ptr{Ptr{UInt8}}(address("completed")),
                            failures=Ptr{Int}(address("failures")))
end

take_completed() =
    Base.unsafe_swap!(get_native_callbacks().completed, Ptr{UInt8}(C_NULL), :acquire)

function free_buffers(buf::Ptr{UInt8})
    while buf != C_NULL
        next = unsafe_load(Ptr{Ptr{UInt8}}(buf + buffer_size))
        Libc.free(buf)
        buf = next
    end
end

take_failures() = Base.unsafe_swap!(get_native_callbacks().failures, 0, :monotonic)

# free the buffers completed outside of a session, e.g. by a session that failed to flush
function discard_buffers()
    take_failures()
    free_buffers(take_completed())
end

# append the completed buffers to `results`, and return how many buffers couldn't be
# allocated
function take_buffers!(results::Vector{ActivityBuffer})
    taken = ActivityBuffer[]
    failures = take_failures()
    buf = take_completed()
    try
        while buf != C_NULL
            trailer = buf + buffer_size
            next = unsafe_load(Ptr{Ptr{UInt8}}(trailer))
            ctx = unsafe_load(Ptr{CUcontext}(trailer + 8))
            stream_id = unsafe_load(Ptr{UInt64}(trailer + 16)) % UInt32
            valid_size = Int(unsafe_load(Ptr{UInt64}(trailer + 24)))
            # this frees the buffer, and counts all of it towards GC pressure
            data = unsafe_wrap(BufferMemory, buf, buffer_size + trailer_size; own=true)
            buf = next
            push!(taken, ActivityBuffer(ctx, stream_id, data, valid_size))
        end
    finally
        free_buffers(buf)   # the buffers that weren't wrapped, if wrapping failed
    end
    # the list is last in, first out
    append!(results, Iterators.reverse(taken))
    return failures
end

"""
    CUPTI.@enable! cfg expr -> value of expr

Macro version of [`CUPTI.enable!(f, cfg)`](@ref CUPTI.enable!) that evaluates `expr`
without creating a closure, similar to `@lock` vs `lock(f, l)`, and returns its value.
"""
macro enable!(cfg, expr)
    mod = @__MODULE__
    quote
        _cfg = $(esc(cfg))
        if _cfg isa $mod.ActivityConfig
            @lock $mod.activity_lock begin
                $mod.discard_buffers()
                callbacks = $mod.get_native_callbacks()
                $mod.cuptiActivityRegisterCallbacks(callbacks.request, callbacks.complete)

                # enable the requested activity kinds. which kinds a CUPTI supports
                # depends on the version and the platform -- Tegra's CUPTI rejects the
                # NVTX marker-data records up to and including CUDA 11 -- and it reports
                # an unsupported kind as CUPTI_ERROR_NOT_COMPATIBLE. Skip those instead
                # of failing the entire profiling session, and only disable what we
                # actually enabled.
                enabled_kinds = $mod.CUpti_ActivityKind[]
                try
                    for activity_kind in _cfg.activity_kinds
                        try
                            $mod.cuptiActivityEnable(activity_kind)
                        catch err
                            if isa(err, $mod.CUPTIError) &&
                               err.code == $mod.CUPTI_ERROR_NOT_COMPATIBLE
                                @debug "CUPTI does not support $activity_kind; not profiling it" _group=:CUDA
                                continue
                            end
                            rethrow()
                        end
                        push!(enabled_kinds, activity_kind)
                    end

                    $(esc(expr))
                finally
                    try
                        for activity_kind in enabled_kinds
                            $mod.cuptiActivityDisable(activity_kind)
                        end
                    finally
                        # flush all activity records, even incomplete ones
                        try
                            $mod.cuptiActivityFlushAll($mod.CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)
                        finally
                            failures = $mod.take_buffers!(_cfg.results)
                            if failures > 0
                                @warn "Could not allocate $failures CUPTI activity buffers; some activity was not recorded."
                            end
                        end
                    end
                end
            end
        elseif _cfg isa $mod.CallbackConfig
            @lock $mod.callback_lock begin
                callback_ptr =
                    @cfunction($mod.callback, Cvoid,
                               (Ptr{Cvoid}, CUpti_CallbackDomain, CUpti_CallbackId, Ptr{Cvoid}))

                GC.@preserve _cfg begin
                    # set-up subscriber
                    subscriber_ref = Ref{CUpti_SubscriberHandle}()
                    $mod.cuptiSubscribe(subscriber_ref, callback_ptr, Base.pointer_from_objref(_cfg))
                    subscriber = subscriber_ref[]

                    # enable domains
                    for callback_kind in _cfg.callback_kinds
                        $mod.cuptiEnableDomain(true, subscriber, callback_kind)
                    end

                    ret = try
                        $(esc(expr))
                    catch
                        # a failed session can break unsubscribing (e.g. PC sampling with
                        # CUPTI 13.0 when another process holds the profiling hardware),
                        # which shouldn't hide the actual error
                        for callback_kind in _cfg.callback_kinds
                            $mod.unchecked_cuptiEnableDomain(false, subscriber, callback_kind)
                        end
                        $mod.unchecked_cuptiUnsubscribe(subscriber)
                        rethrow()
                    end
                    for callback_kind in _cfg.callback_kinds
                        $mod.cuptiEnableDomain(false, subscriber, callback_kind)
                    end
                    $mod.cuptiUnsubscribe(subscriber)
                    ret
                end
            end
        else
            error("@enable! expects an ActivityConfig or CallbackConfig")
        end
    end
end

"""
    CUPTI.enable!(f, cfg::Union{CUPTI.ActivityConfig,CUPTI.CallbackConfig}) -> result of f()

Enable the activities or callbacks configured by `cfg` while calling `f()`, and return
its result.

For an [`ActivityConfig`](@ref), CUPTI's activity records are flushed into `cfg` when `f`
returns (`cuptiActivityFlushAll`). Records of work that hasn't completed by then are
incomplete or missing, so synchronize at the end of `f`. For a [`CallbackConfig`](@ref), it
subscribes to the configured callback domains (`cuptiSubscribe` and `cuptiEnableDomain`).

Only one configuration of each kind can be enabled at a time: an `enable!` in another task
waits for the previous one to finish, while nesting `enable!` calls for the same kind of
configuration in one task is not supported. CUPTI only supports a single callback
subscriber per process, so a `CallbackConfig` can't be enabled while another tool (e.g. a
profiler) is subscribed.
"""
function enable!(@nospecialize(f::Base.Callable), cfg::ActivityConfig)
    @enable! cfg f()
end

function enable!(@nospecialize(f::Base.Callable), cfg::CallbackConfig)
    @enable! cfg f()
end

"""
    CUPTI.process(f, cfg::CUPTI.ActivityConfig) -> nothing

Call `f(ctx, stream_id, record)` for every activity record collected in `cfg`, in the
order CUPTI delivered them. `ctx` and `stream_id` are the `CuContext` (or `nothing`) and
the CUPTI stream ID that CUPTI reported for the buffer holding the record. `record` is a
copy of the record, as the newest CUPTI struct that the loaded CUDA version writes for its
kind: e.g. `CUpti_ActivityAPI` for driver and runtime API calls, or
`CUpti_ActivityKernel9` for kernels on CUDA 12. Records of kinds this function doesn't
know about are skipped with a warning. Wraps `cuptiActivityGetNextRecord`.
"""
function process(f, cfg::ActivityConfig)
    activity_types = Dict(
        CUPTI_ACTIVITY_KIND_DRIVER              => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_RUNTIME             => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_NAME                => CUpti_ActivityName,
        CUPTI_ACTIVITY_KIND_MARKER              => CUpti_ActivityMarker2,
        CUPTI_ACTIVITY_KIND_MARKER_DATA         => CUpti_ActivityMarkerData,
    )
    # CUPTI emits whichever record version the CUDA release it ships with implements, so
    # select the newest struct that release is known to write. Each new version only
    # appends fields, which makes the direction matter: reading a newer record through an
    # older struct just hides the added fields, but reading an older record through a
    # newer struct would run past the record into the next one. When adding a version,
    # take the introducing release from the CUPTI release notes, and prefer the later
    # release when unsure -- being late costs fields, being early corrupts them.
    #
    # NOTE: the CUPTI version is unreliable (e.g. CUDA 11.5 and 11.6 both had CUPTI 16),
    #       so key off the CUDA version instead.
    cuda = CUDACore.runtime_version()
    ## kernel activities
    activity_types[CUPTI_ACTIVITY_KIND_KERNEL] =
        if cuda >= v"13.4"
            CUpti_ActivityKernel13
        elseif cuda >= v"13.3"
            CUpti_ActivityKernel12
        elseif cuda >= v"13.1"
            CUpti_ActivityKernel11
        elseif cuda >= v"13.0"
            CUpti_ActivityKernel10
        elseif cuda >= v"12.0"
            CUpti_ActivityKernel9
        elseif cuda >= v"11.8"
            CUpti_ActivityKernel8
        elseif cuda >= v"11.6"
            CUpti_ActivityKernel7
        elseif cuda >= v"11.2"
            CUpti_ActivityKernel6
        elseif cuda >= v"11.1"
            CUpti_ActivityKernel5
        else # oldest supported record layout
            CUpti_ActivityKernel4
        end
    activity_types[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL] =
        activity_types[CUPTI_ACTIVITY_KIND_KERNEL]
    ## memcpy activities
    activity_types[CUPTI_ACTIVITY_KIND_MEMCPY] =
        if cuda >= v"13.4"
            CUpti_ActivityMemcpy7
        elseif cuda >= v"12.8"
            CUpti_ActivityMemcpy6
        elseif cuda >= v"11.6"
            CUpti_ActivityMemcpy5
        elseif cuda >= v"11.1"
            CUpti_ActivityMemcpy4
        elseif cuda >= v"11.0"
            CUpti_ActivityMemcpy3
        else # oldest supported record layout
            CUpti_ActivityMemcpy
        end
    activity_types[CUPTI_ACTIVITY_KIND_MEMSET] =
        if cuda >= v"13.4"
            CUpti_ActivityMemset5
        elseif cuda >= v"11.6"
            CUpti_ActivityMemset4
        elseif cuda >= v"11.1"
            CUpti_ActivityMemset3
        elseif cuda >= v"11.0"
            CUpti_ActivityMemset2
        else # oldest supported record layout
            CUpti_ActivityMemset
        end
    activity_types[CUPTI_ACTIVITY_KIND_MEMORY2] =
        if cuda >= v"12.6"
            CUpti_ActivityMemory4
        elseif cuda >= v"11.6"
            CUpti_ActivityMemory3
        elseif cuda >= v"11.2"
            CUpti_ActivityMemory2
        else # oldest supported record layout
            CUpti_ActivityMemory
        end

    # extract typed activity records
    for buf in cfg.results
        ctx = buf.ctx == C_NULL ? nothing : unsafe_wrap(CuContext, buf.ctx)
        stream_id, data, valid_sz = buf.stream_id, buf.data, buf.valid_size
        # XXX: can we reconstruct the stream from the stream ID?

        record_ptr = Ref{Ptr{CUpti_Activity}}(C_NULL)
        buf_ptr = pointer(data)
        GC.@preserve data while true
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
