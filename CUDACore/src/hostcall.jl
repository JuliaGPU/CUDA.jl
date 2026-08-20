# Hostcall: host-side service
#
# The device-side protocol lives in `device/intrinsics/hostcall.jl`. This file provides the
# pinned memory area that backs it (one per context, created lazily), the target registry,
# and the server: a single foreign libuv thread that sweeps every area while kernels that
# may hostcall are in flight, and wakes up periodically otherwise. The server thread is
# independent of Julia's scheduler, so hostcalls keep being serviced while the launching
# thread is blocked in a CUDA call.

export HostFunction
@public HostcallException, HostcallArea, hostcall_area, hostcall_drain, hostcall_available

using Preferences: @load_preference


## preferences

# whether hostcalls are enabled at all
const hostcall_enabled = @load_preference("hostcall", true)::Bool

# the number of ports per area; defaults to the number of resident warps of the device
const hostcall_ports_pref = @load_preference("hostcall_ports", nothing)

# hostcalls require the host to poll while a kernel is running. on Windows with WDDM
# kernels are batched and subject to a 2s watchdog, so require an explicit opt-in there.
const hostcall_wddm = @load_preference("hostcall_wddm", false)::Bool

"""
    hostcall_available([dev::CuDevice]) -> Bool

Whether hostcalls (and thus hostcall-based functionality such as device exception
reporting) are available on `dev`. Controlled by the `hostcall` preference; on Windows
hostcalls are only enabled with the TCC driver or the `hostcall_wddm` preference.
"""
function hostcall_available(dev::CuDevice=device())
    hostcall_enabled || return false
    if Sys.iswindows() && !hostcall_wddm
        return attribute(dev, DEVICE_ATTRIBUTE_TCC_DRIVER) == 1
    end
    return true
end


## area

"""
    HostcallArea

Host-side owner of a hostcall area: pinned, device-mapped memory for the mailboxes,
headers and packets of `nports` ports, the lock bitfield in device memory, and the
`HostcallClient` handed to kernels. Areas are created per context by [`hostcall_area`](@ref).
"""
mutable struct HostcallArea
    const ctx::CuContext
    const nports::Int
    const mem::HostMemory
    const base::Ptr{UInt8}
    const layout::@NamedTuple{inbox::Int, outbox::Int, header::Int, packet::Int, total::Int}
    const locks::DeviceMemory
    const client::HostcallClient

    # host-side sweep state
    const shadow::Vector{UInt32}        # the inbox words we last wrote (host is the only writer)
    const outbox::Vector{UInt32}        # view of the pinned outbox array
    cursor::Int                         # sweeps resume after the last serviced port

    # handlers run on a dedicated non-blocking stream, so that they never wait on the
    # stream the calling kernel is running on
    stream::Union{Nothing,CuStream}
end

const HOSTCALL_SWEEP_CHUNK = 64
const HOSTCALL_MIN_PORTS = 64
const HOSTCALL_MAX_PORTS = 4096

function HostcallArea(ctx::CuContext, nports::Integer)
    nports > 0 || throw(ArgumentError("the number of hostcall ports must be positive"))
    nports <= typemax(UInt32) - (HOSTCALL_SWEEP_CHUNK - 1) ||
        throw(ArgumentError("too many hostcall ports: $nports"))
    nports = cld(Int(nports), HOSTCALL_SWEEP_CHUNK) * HOSTCALL_SWEEP_CHUNK
    layout = hostcall_packet_layout(nports)
    context!(ctx) do
        mem = alloc(HostMemory, layout.total, MEMHOSTALLOC_DEVICEMAP | MEMHOSTALLOC_PORTABLE)
        base = convert(Ptr{UInt8}, mem)
        # the protocol assumes a unified address space: the same pointer on both sides
        UInt(base) == UInt(convert(CuPtr{UInt8}, mem)) ||
            error("Hostcall requires unified addressing (host and device pointers to pinned memory differ)")
        unsafe_wrap(Array, base, layout.total) .= 0
        locks = alloc(DeviceMemory, 4 * cld(nports, 32))
        cuMemsetD32_v2(locks, 0, cld(nports, 32))
        dp(off, T) = reinterpret(LLVMPtr{T,AS.Global}, base + off)
        client = HostcallClient(nports, dp(layout.inbox, UInt32), dp(layout.outbox, UInt32),
                                dp(layout.header, HostcallHeader), dp(layout.packet, UInt8),
                                reinterpret(LLVMPtr{UInt32,AS.Global}, convert(CuPtr{UInt32}, locks)))
        shadow = zeros(UInt32, nports)
        outbox = unsafe_wrap(Array, convert(Ptr{UInt32}, base + layout.outbox), nports)
        HostcallArea(ctx, nports, mem, base, layout, locks, client, shadow, outbox, 0, nothing)
    end
end

inbox_ptr(a::HostcallArea, i) = convert(Ptr{UInt32}, a.base + a.layout.inbox) + 4i
outbox_ptr(a::HostcallArea, i) = convert(Ptr{UInt32}, a.base + a.layout.outbox) + 4i
header_ptr(a::HostcallArea, i) =
    convert(Ptr{HostcallHeader}, a.base + a.layout.header) + sizeof(HostcallHeader) * i
packet_ptr(a::HostcallArea, i, lane) =
    a.base + a.layout.packet + (i * HOSTCALL_LANES + lane) * HOSTCALL_PACKET_SIZE

mailbox_load(p::Ptr{UInt32}) = Core.Intrinsics.atomic_pointerref(p, :acquire)
mailbox_store!(p::Ptr{UInt32}, v::UInt32) = Core.Intrinsics.atomic_pointerset(p, v, :release)

function hostcall_stream(a::HostcallArea)
    s = a.stream
    if s === nothing
        s = context!(a.ctx) do
            CuStream(; flags=STREAM_NON_BLOCKING)
        end
        a.stream = s
    end
    return s
end

# the default number of ports: enough for every resident warp, so that a warp never has to
# wait for another warp to release a port (GPUs have no forward-progress guarantee)
function hostcall_default_ports(dev::CuDevice)
    if hostcall_ports_pref !== nothing
        return Int(hostcall_ports_pref)
    end
    sms = attribute(dev, DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    warps = attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) ÷ 32
    return clamp(sms * warps, HOSTCALL_MIN_PORTS, HOSTCALL_MAX_PORTS)
end

# all areas, as an immutable snapshot that the server thread can read without locking
mutable struct HostcallAreas
    Base.@atomic list::Vector{HostcallArea}
end
const hostcall_areas = HostcallAreas(HostcallArea[])
const hostcall_areas_lock = ReentrantLock()

"""
    hostcall_area(ctx::CuContext=context(); ports=HOSTCALL_MIN_PORTS) -> HostcallArea

Return the hostcall area of `ctx` with at least `ports` ports, creating it (and starting
the server thread) if necessary. Kernels that do not call host functions only need a small
area (for exception reporting), so contexts start with a minimal one and get a larger area
when a kernel that uses hostcalls is linked; earlier areas stay alive and keep being
serviced, because running kernels may still refer to them.
"""
function hostcall_area(ctx::CuContext=context(); ports::Integer=HOSTCALL_MIN_PORTS)
    @lock hostcall_areas_lock begin
        areas = Base.@atomic hostcall_areas.list
        # newest first: the last area created for a context is the largest
        for area in Iterators.reverse(areas)
            area.ctx == ctx && area.nports >= ports && return area
        end
        area = HostcallArea(ctx, ports)
        hostcall_server_start()
        Base.@atomic hostcall_areas.list = [areas..., area]
        return area
    end
end

"""
    hostcall_client(ctx::CuContext) -> HostcallClient

The client descriptor kernels should be launched with: the newest area of `ctx`, or an
all-null client when hostcalls are not available.
"""
function hostcall_client(ctx::CuContext, dev::CuDevice=device(ctx); ports::Integer=HOSTCALL_MIN_PORTS)
    hostcall_available(dev) || return HostcallClient()
    return hostcall_area(ctx; ports).client
end


## target registry

# a registered host function and how its arguments and result are encoded
struct HostcallTarget
    key::Any                 # static registry key, or nothing for a runtime handle
    f::Any
    RT::Type                # return type (`Nothing` for calls without a reply)
    AT::Type                # payload tuple type; includes `f` when it is not stored
    stored::Bool            # callable is stored in `f`, rather than included in the payload
end

const hostcall_targets = Dict{UInt64,HostcallTarget}()
const hostcall_targets_lock = Threads.SpinLock()

"""
    register_hostcall_targets!(targets)

Register the statically-known hostcall targets of a kernel (pairs of identifier and key
type `Tuple{F,RT,AT}`, as recorded by the compiler) so that the server can dispatch calls.
"""
function register_hostcall_targets!(targets)
    isempty(targets) && return
    for (id, K) in targets
        F, RT, AT = K.parameters
        stored = Base.issingletontype(F)
        f = stored ? F.instance : nothing
        target = HostcallTarget(K, f, RT, AT, stored)

        # Compile the handler on the registering thread, before taking the registry lock.
        precompile(stored ? Tuple{F, AT.parameters...} : AT)
        @lock hostcall_targets_lock begin
            previous = get(hostcall_targets, id, nothing)
            if previous === nothing
                hostcall_targets[id] = target
            elseif previous.key !== K
                error("hostcall target identifier collision between $(previous.key) and $K")
            end
        end
    end
    return
end

hostcall_target(id::UInt64) = @lock hostcall_targets_lock get(hostcall_targets, id, nothing)

"""
    HostFunction(f, RT, AT::Type{<:Tuple}) -> HostFunction{RT,AT}

Register `f` as a host function callable from device code with arguments of type `AT`,
returning a value of type `RT`. Pass the handle to a kernel; it becomes a
[`DeviceHostFunction`](@ref) that can be called like a function, or used with
[`@hostcall`](@ref). Use this for closures and other callables whose value cannot be
recovered from their type, or when the target is only known at run time. The handle is
registered until `close` is called. Call `close` only after kernels using the handle have
completed; a launched kernel retains the numeric device handle, not the Julia object.

```julia
data = rand(Float32, 10)
hf = HostFunction(i -> data[i], Float32, Tuple{Int})
@cuda kernel(out, hf)    # kernel(out, hf) = (out[threadIdx().x] = hf(threadIdx().x); nothing)
```
"""
mutable struct HostFunction{RT,AT}
    const f::Any
    const id::UInt64
    registered::Bool

    function HostFunction{RT,AT}(f) where {RT,AT<:Tuple}
        isbitstype(RT) || RT === Nothing ||
            throw(ArgumentError("hostcall return types must be isbits or Nothing, got $RT"))
        isconcretetype(AT) && isbitstype(AT) ||
            throw(ArgumentError("hostcall argument types must be a concrete tuple of isbits types, got $AT"))
        id = next_hostfunction_id()
        hf = new{RT,AT}(f, id, true)
        @lock hostcall_targets_lock begin
            hostcall_targets[id] = HostcallTarget(nothing, f, RT, AT, true)
        end
        return hf
    end
end
HostFunction(f, ::Type{RT}, ::Type{AT}) where {RT,AT<:Tuple} = HostFunction{RT,AT}(f)

const hostfunction_counter = Threads.Atomic{UInt64}(0)
function next_hostfunction_id()
    n = Threads.atomic_add!(hostfunction_counter, UInt64(1))
    n < HOSTCALL_STATIC_ID_MASK || error("hostcall handle identifier space exhausted")
    return HOSTCALL_RUNTIME_ID_BIT | (n + 1)
end

function Base.close(hf::HostFunction)
    @lock hostcall_targets_lock begin
        hf.registered || return
        hf.registered = false
        delete!(hostcall_targets, hf.id)
    end
    return
end

Adapt.adapt_storage(::KernelAdaptor, hf::HostFunction{RT,AT}) where {RT,AT} =
    DeviceHostFunction{RT,AT}(hf.id)

# conversion of values received from the device (pointers already arrive as `CuPtr`, see
# the device-side `hostconvert`)
hostconvert_host(x) = x
hostconvert_host(p::LLVMPtr{T}) where {T} = reinterpret(CuPtr{T}, p)


## exceptions and deferred output

"""
    HostcallException

Thrown by `synchronize()` when a host function called from a kernel threw, when the
target of a call is unknown, or when its result could not be converted. The calling device
thread has been stopped. The original error and backtrace are available in the `error`
and `backtrace` fields.
"""
struct HostcallException <: Exception
    target::Any
    error::Any
    backtrace::Any
end

function Base.showerror(io::IO, err::HostcallException)
    print(io, "HostcallException: error while servicing a hostcall to ", err.target, ":\n")
    showerror(io, err.error)
    if err.backtrace !== nothing
        println(io)
        Base.show_backtrace(io, err.backtrace)
    end
end

const hostcall_exceptions = HostcallException[]
const hostcall_exceptions_lock = Threads.SpinLock()
const hostcall_exceptions_pending = Threads.Atomic{Int}(0)

function record_hostcall_exception!(target, err, bt=nothing)
    @lock hostcall_exceptions_lock push!(hostcall_exceptions, HostcallException(target, err, bt))
    Threads.atomic_add!(hostcall_exceptions_pending, 1)
    return
end

# throw the oldest pending exception (called from `check_exceptions`)
function check_hostcall_exceptions()
    hostcall_exceptions_pending[] == 0 && return
    err = @lock hostcall_exceptions_lock begin
        isempty(hostcall_exceptions) && return
        Threads.atomic_sub!(hostcall_exceptions_pending, 1)
        popfirst!(hostcall_exceptions)
    end
    throw(err)
end

# output from the print family is not written by the server thread (libuv I/O from a
# foreign thread can hang while the main thread is blocked in a ccall, julia#55525), but
# queued and emitted by a Julia task, or by whoever drains the hostcall area next.
const hostcall_output = IOBuffer()
const hostcall_output_lock = Threads.SpinLock()
const hostcall_output_pending = Threads.Atomic{Int}(0)
const hostcall_output_cond = Ref{Union{Nothing,Base.AsyncCondition}}(nothing)
const hostcall_output_cond_lock = ReentrantLock()

is_print_target(@nospecialize(f)) =
    f === print || f === println || f === printstyled || f === show || f === display

function queue_hostcall_output(f, args...)
    io = IOBuffer()
    if f === display
        show(io, MIME"text/plain"(), args...)
        println(io)
    else
        f(io, args...)
    end
    data = take!(io)
    @lock hostcall_output_lock begin
        write(hostcall_output, data)
    end
    Threads.atomic_add!(hostcall_output_pending, 1)
    cond = hostcall_output_cond[]
    if cond !== nothing
        # wakes the printer task, if thread 1 is free to run it; otherwise the output is
        # flushed by the next synchronization
        ccall(:uv_async_send, Cint, (Ptr{Cvoid},), cond)
    end
    return
end

function flush_hostcall_output()
    hostcall_output_pending[] == 0 && return
    data = @lock hostcall_output_lock begin
        hostcall_output_pending[] = 0
        take!(hostcall_output)
    end
    isempty(data) || write(stdout, data)
    return
end

function hostcall_output_printer()
    @lock hostcall_output_cond_lock begin
        if hostcall_output_cond[] === nothing
            hostcall_output_cond[] = Base.AsyncCondition() do _
                flush_hostcall_output()
            end
        end
    end
end


## servicing a port

# host-side view of a single in-progress call on a port: the server owns the buffer
# whenever the outbox differs from the inbox it last wrote
mutable struct HostPort
    area::HostcallArea
    index::Int
    out::UInt32         # last observed outbox value
    in::UInt32          # last written inbox value (bit 0 is the ownership bit)
end

# give the buffer to the device (optionally with data written to the packets) and report a
# status; the next device flip makes it ours again
function hport_flip!(p::HostPort, status::UInt32=HOSTCALL_STATUS_OK)
    p.in = p.out | (status << 1)
    @inbounds p.area.shadow[p.index + 1] = p.in
    mailbox_store!(inbox_ptr(p.area, p.index), p.in)
    return
end

# wait for the device to flip its outbox after we handed the buffer back. the device is
# actively waiting on us when this is called, so this should be quick; time out in case
# the kernel died.
function hport_wait!(p::HostPort; timeout=10.0)
    t0 = time()
    while true
        out = mailbox_load(outbox_ptr(p.area, p.index))
        if out != (p.in & UInt32(1))
            p.out = out
            return true
        end
        ccall(:jl_cpu_pause, Cvoid, ())
        ccall(:jl_gc_safepoint, Cvoid, ())
        time() - t0 > timeout && return false
    end
end

lane_packet(p::HostPort, lane) = packet_ptr(p.area, p.index, lane)

# read a value of type `T` from a lane's packets, receiving additional chunks if needed
function read_lanes(p::HostPort, ::Type{T}, mask::UInt32) where {T}
    values = Vector{T}(undef, 32)
    if sizeof(T) <= HOSTCALL_PACKET_SIZE
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds values[lane + 1] = unsafe_load(convert(Ptr{T}, lane_packet(p, lane)))
        end
    else
        nchunks = cld(sizeof(T), HOSTCALL_PACKET_SIZE)
        bufs = [Vector{UInt8}(undef, nchunks * HOSTCALL_PACKET_SIZE) for _ in 1:32]
        for chunk in 0:nchunks-1
            if chunk > 0
                hport_flip!(p)
                hport_wait!(p) || error("timed out receiving hostcall arguments")
            end
            for lane in 0:31
                (mask >> lane) & 1 == 0 && continue
                unsafe_copyto!(pointer(bufs[lane + 1]) + chunk * HOSTCALL_PACKET_SIZE,
                               lane_packet(p, lane), HOSTCALL_PACKET_SIZE)
            end
        end
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds values[lane + 1] = unsafe_load(convert(Ptr{T}, pointer(bufs[lane + 1])))
        end
    end
    return values
end

# write each lane's value of type `T` to its packets and hand the buffer back; for large
# values this involves multiple flips, for which the device must be receiving.
function write_lanes!(p::HostPort, values::Vector{T}, mask::UInt32, status::UInt32) where {T}
    if sizeof(T) <= HOSTCALL_PACKET_SIZE
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds unsafe_store!(convert(Ptr{T}, lane_packet(p, lane)), values[lane + 1])
        end
        hport_flip!(p, status)
    else
        nchunks = cld(sizeof(T), HOSTCALL_PACKET_SIZE)
        bufs = [Vector{UInt8}(undef, nchunks * HOSTCALL_PACKET_SIZE) for _ in 1:32]
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds unsafe_store!(convert(Ptr{T}, pointer(bufs[lane + 1])), values[lane + 1])
        end
        for chunk in 0:nchunks-1
            if chunk > 0
                hport_wait!(p) || error("timed out sending hostcall results")
            end
            for lane in 0:31
                (mask >> lane) & 1 == 0 && continue
                unsafe_copyto!(lane_packet(p, lane),
                               pointer(bufs[lane + 1]) + chunk * HOSTCALL_PACKET_SIZE,
                               HOSTCALL_PACKET_SIZE)
            end
            hport_flip!(p, status)
        end
    end
    return
end

# invoke a registered target for every live lane of a port
function service_target!(p::HostPort, target::HostcallTarget, hdr::HostcallHeader)
    mask = hdr.mask
    RT = target.RT
    reply = (hdr.flags & HOSTCALL_FLAG_ASYNC) == 0 && RT !== Nothing
    payloads = read_lanes(p, target.AT, mask)
    results = reply ? Vector{RT}(undef, 32) : nothing
    status = HOSTCALL_STATUS_OK
    for lane in 0:31
        (mask >> lane) & 1 == 0 && continue
        payload = @inbounds payloads[lane + 1]
        f, args = if target.stored
            target.f, payload
        else
            first(payload), Base.tail(payload)
        end
        args = map(hostconvert_host, args)
        try
            if is_print_target(f)
                queue_hostcall_output(f, args...)
                rv = nothing
            else
                rv = Base.invokelatest(f, args...)
            end
            if reply
                results[lane + 1] = convert(RT, rv)
            end
        catch err
            record_hostcall_exception!(f, err, catch_backtrace())
            status = HOSTCALL_STATUS_ERROR
        end
    end
    if status != HOSTCALL_STATUS_OK || !reply
        hport_flip!(p, status)
    else
        write_lanes!(p, results, mask, status)
    end
    return
end

# built-in targets are registered in `hostcall_builtins` by the code that implements them
const hostcall_builtins = Dict{UInt64,Function}()

function service_port!(a::HostcallArea, i::Int, out::UInt32)
    hdr = unsafe_load(header_ptr(a, i))
    p = HostPort(a, i, out, @inbounds a.shadow[i + 1])
    context!(a.ctx) do
        stream!(hostcall_stream(a)) do
            if hdr.target < HOSTCALL_BUILTIN_IDS
                handler = get(hostcall_builtins, hdr.target, nothing)
                if handler === nothing
                    record_hostcall_exception!(hdr.target, ErrorException("unknown built-in hostcall target"))
                    hport_flip!(p, HOSTCALL_STATUS_ERROR)
                else
                    handler(p, hdr)
                end
            else
                target = hostcall_target(hdr.target)
                if target === nothing
                    record_hostcall_exception!(hdr.target, ErrorException("unknown hostcall target; was the kernel compiled in another session without its targets being registered?"))
                    hport_flip!(p, HOSTCALL_STATUS_ERROR)
                else
                    service_target!(p, target, hdr)
                end
            end
        end
    end
    return
end

# one sweep over the ports of an area; returns the number of ports serviced. the outbox
# words are compared against our shadow of the inbox in chunks, which vectorizes and keeps
# the cost of scanning idle ports negligible.
function sweep!(a::HostcallArea)
    n = 0
    nports = a.nports
    outbox = a.outbox
    shadow = a.shadow
    start = (a.cursor ÷ HOSTCALL_SWEEP_CHUNK) * HOSTCALL_SWEEP_CHUNK
    @inbounds for c0 in 0:HOSTCALL_SWEEP_CHUNK:nports-1
        cbase = (start + c0) % nports
        cend = cbase + HOSTCALL_SWEEP_CHUNK - 1
        d = UInt32(0)
        @simd for j in cbase:cend
            d |= (outbox[j+1] ⊻ shadow[j+1]) & UInt32(1)
        end
        d == 0 && continue
        for j in cbase:cend
            out = mailbox_load(outbox_ptr(a, j))
            out == (shadow[j+1] & UInt32(1)) && continue
            service_port!(a, j, out)
            n += 1
            a.cursor = j + 1 >= nports ? 0 : j + 1
        end
    end
    return n
end


## server thread

mutable struct HostcallServer
    const mutex::Ptr{Cvoid}         # uv_mutex_t, guards `cond`
    const cond::Ptr{Cvoid}          # uv_cond_t, signalled when a kernel is armed
    const sem::Ptr{Cvoid}           # uv_sem_t, posted by `cuLaunchHostFunc` after every armed kernel
    const sweep_lock::Ptr{Cvoid}    # uv_mutex_t, serializes sweeps (server thread vs. draining tasks)
    const launches::Threads.Atomic{Int}
    const sweep_owner::Threads.Atomic{Int}   # thread id holding `sweep_lock`, or 0
    finished::Int                   # launches that completed; only touched by the server thread
end

const hostcall_server = Ref{Union{Nothing,HostcallServer}}(nothing)
const hostcall_server_lock = ReentrantLock()

# acquire the sweep lock without blocking the thread in a GC-unsafe state
function lock_sweeps(srv::HostcallServer)
    while @ccall(uv_mutex_trylock(srv.sweep_lock::Ptr{Cvoid})::Cint) != 0
        ccall(:jl_cpu_pause, Cvoid, ())
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
    srv.sweep_owner[] = Threads.threadid()
    return
end
function unlock_sweeps(srv::HostcallServer)
    srv.sweep_owner[] = 0
    @ccall uv_mutex_unlock(srv.sweep_lock::Ptr{Cvoid})::Cvoid
    return
end
# whether the current thread is in the middle of a sweep, i.e. running a handler
sweeping(srv::HostcallServer) = srv.sweep_owner[] == Threads.threadid()
function hostcall_in_handler()
    srv = hostcall_server[]
    srv === nothing && return false
    return sweeping(srv)
end

function sweep_all!()
    n = 0
    for area in Base.@atomic hostcall_areas.list
        n += sweep!(area)
    end
    return n
end

# wait for a kernel to be armed, or for the heartbeat interval to pass. a thread blocked in
# a plain ccall holds up garbage collection, so the wait is GC-safe (and short anyway).
const HOSTCALL_HEARTBEAT_NS = 1_000_000
function hostcall_server_wait(srv::HostcallServer)
    @ccall uv_mutex_lock(srv.mutex::Ptr{Cvoid})::Cvoid
    if srv.launches[] - srv.finished == 0
        @gcsafe_ccall uv_cond_timedwait(srv.cond::Ptr{Cvoid}, srv.mutex::Ptr{Cvoid},
                                        HOSTCALL_HEARTBEAT_NS::UInt64)::Cint
    end
    @ccall uv_mutex_unlock(srv.mutex::Ptr{Cvoid})::Cvoid
    return
end

const HOSTCALL_SPIN_BEFORE_SLEEP = 1 << 16    # empty sweeps before backing off to sleeps

function hostcall_server_main(srv::HostcallServer)
    idle = 0
    while true
        try
            # account for completed launches
            while @ccall(uv_sem_trywait(srv.sem::Ptr{Cvoid})::Cint) == 0
                srv.finished += 1
            end
            armed = srv.launches[] - srv.finished

            lock_sweeps(srv)
            found = try
                sweep_all!()
            finally
                unlock_sweeps(srv)
            end

            if armed > 0
                # kernels that may hostcall are running: poll, backing off to short sleeps
                # when nothing has happened for a while
                if found > 0
                    idle = 0
                else
                    idle += 1
                    if idle < HOSTCALL_SPIN_BEFORE_SLEEP
                        ccall(:jl_cpu_pause, Cvoid, ())
                    else
                        @gcsafe_ccall usleep(20::Cuint)::Cint
                    end
                end
            else
                # idle: heartbeat sweeps, woken early by the next launch
                idle = 0
                hostcall_server_wait(srv)
            end
        catch err
            # errors in the server itself (not in handlers, which are caught separately)
            # are reported at the next synchronization; keep the thread alive
            record_hostcall_exception!(:server, err, catch_backtrace())
        end
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
end

function hostcall_server_entry(::Ptr{Cvoid})
    srv = hostcall_server[]::HostcallServer
    hostcall_server_main(srv)
    return nothing
end


function hostcall_server_start()
    @lock hostcall_server_lock begin
        hostcall_server[] === nothing || return
        # libuv synchronization objects; never freed
        mutex = Libc.malloc(64)
        cond = Libc.malloc(64)
        sem = Libc.malloc(64)
        sweep_lock = Libc.malloc(64)
        @ccall(uv_mutex_init(mutex::Ptr{Cvoid})::Cint) == 0 || error("uv_mutex_init failed")
        @ccall(uv_mutex_init(sweep_lock::Ptr{Cvoid})::Cint) == 0 || error("uv_mutex_init failed")
        @ccall(uv_cond_init(cond::Ptr{Cvoid})::Cint) == 0 || error("uv_cond_init failed")
        @ccall(uv_sem_init(sem::Ptr{Cvoid}, 0::Cuint)::Cint) == 0 || error("uv_sem_init failed")
        srv = HostcallServer(mutex, cond, sem, sweep_lock, Threads.Atomic{Int}(0),
                             Threads.Atomic{Int}(0), 0)

        # Compile the server and its dynamically-dispatched handlers on this thread.
        precompile(hostcall_server_main, (HostcallServer,))
        precompile(service_port!, (HostcallArea, Int, UInt32))
        for handler in values(hostcall_builtins)
            precompile(handler, (HostPort, HostcallHeader))
        end

        # the thread is adopted by Julia when it first calls into the @cfunction, and is
        # never torn down
        hostcall_server[] = srv
        tid = Ref{NTuple{32, UInt8}}(ntuple(i -> 0x0, 32))
        cb = @cfunction(hostcall_server_entry, Cvoid, (Ptr{Cvoid},))
        err = @ccall uv_thread_create(tid::Ptr{Cvoid}, cb::Ptr{Cvoid}, C_NULL::Ptr{Cvoid})::Cint
        if err != 0
            hostcall_server[] = nothing
            Base.uv_error("uv_thread_create", err)
        end
        err = @ccall uv_thread_detach(tid::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_detach", err)

        # the printer task is created on a Julia thread
        hostcall_output_printer()
    end
    return
end


## arming and draining

"""
    hostcall_arm!()

Announce an upcoming launch of a kernel that may hostcall: the server starts polling.
Balanced by [`hostcall_disarm!`](@ref) after the kernel, enqueued on its stream.
"""
function hostcall_arm!()
    srv = hostcall_server[]::HostcallServer
    Threads.atomic_add!(srv.launches, 1)
    @ccall uv_mutex_lock(srv.mutex::Ptr{Cvoid})::Cvoid
    @ccall uv_cond_signal(srv.cond::Ptr{Cvoid})::Cvoid
    @ccall uv_mutex_unlock(srv.mutex::Ptr{Cvoid})::Cvoid
    return
end

"""
    hostcall_disarm!(stream::CuStream)
    hostcall_disarm!()

Mark a launch as completed, either immediately or when `stream` reaches this point. The
latter enqueues a plain C callback (`uv_sem_post`) that only touches a semaphore, so it
neither calls into Julia nor needs the libuv event loop.
"""
function hostcall_disarm!(stream::CuStream)
    srv = hostcall_server[]::HostcallServer
    cuLaunchHostFunc(stream, cglobal(:uv_sem_post), srv.sem)
    return
end
function hostcall_disarm!()
    srv = hostcall_server[]::HostcallServer
    @ccall uv_sem_post(srv.sem::Ptr{Cvoid})::Cvoid
    return
end

"""
    hostcall_drain()

Service every pending hostcall and flush deferred output. Called by `synchronize()` and
`device_synchronize()`, so that asynchronous hostcalls made by a kernel have completed by
the time those return; handlers may run on the calling task.
"""
# whether any port of an area is owned by the host, i.e. has a pending call; a lock-free
# read that may race with a sweep, in which case the lock-protected sweep decides
function pending_ports(a::HostcallArea)
    outbox = a.outbox
    shadow = a.shadow
    d = UInt32(0)
    @inbounds @simd for j in 1:a.nports
        d |= (outbox[j] ⊻ shadow[j]) & UInt32(1)
    end
    return d != 0
end

function hostcall_drain()
    srv = hostcall_server[]
    srv === nothing && return
    areas = Base.@atomic hostcall_areas.list
    isempty(areas) && return
    # fast path: nothing to service or flush
    if !any(pending_ports, areas) && hostcall_output_pending[] == 0
        return
    end
    if sweeping(srv)
        # called from a handler (e.g. a handler synchronizing the hostcall stream): the
        # port being serviced is still pending, so a nested sweep would re-enter the
        # handler. handlers cannot wait for other hostcalls anyway.
        return
    end
    lock_sweeps(srv)
    try
        sweep_all!()
    finally
        unlock_sweeps(srv)
    end
    flush_hostcall_output()
    return
end

# forget the areas of a context that is about to be destroyed: its pinned memory goes away
# with it, so the server must stop sweeping it. (the hostcall machinery otherwise assumes
# that contexts are immortal, as CUDA.jl's primary contexts are.)
function hostcall_forget!(pred)
    srv = hostcall_server[]
    srv === nothing && return
    nested = sweeping(srv)
    nested || lock_sweeps(srv)
    try
        @lock hostcall_areas_lock begin
            areas = Base.@atomic hostcall_areas.list
            Base.@atomic hostcall_areas.list = filter(a -> !pred(a), areas)
        end
    finally
        nested || unlock_sweeps(srv)
    end
    return
end
hostcall_forget!(ctx::CuContext) = hostcall_forget!(a -> a.ctx == ctx)
hostcall_forget!(dev::CuDevice) = hostcall_forget!(a -> isvalid(a.ctx) && device(a.ctx) == dev)
