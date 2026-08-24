# Hostcall: host-side service
#
# The device-side protocol lives in `device/intrinsics/hostcall.jl`. This file provides the
# pinned memory area that backs it (one per context, created lazily), the target registry,
# and the server: a single foreign libuv thread that sweeps every area while kernels that
# may hostcall are in flight, and wakes up periodically otherwise. The server thread is
# independent of Julia's scheduler, so hostcalls keep being serviced while the launching
# thread is blocked in a CUDA call.

@public HostcallException, HostcallArea, hostcall_area, hostcall_drain

using Preferences: @load_preference


## preferences

# the number of ports per area; defaults to the number of resident warps of the device
const hostcall_ports_pref = @load_preference("hostcall_ports", nothing)


## area

"""
    HostcallArea

Host-side owner of a hostcall area: pinned, device-mapped memory for the mailboxes,
headers and packets of `nports` ports, the lock bitfield in device memory, and the mapped
`HostcallClient` descriptor referenced by kernels. Exception-only areas are polled by the
server heartbeat; other areas are polled while an ordinary hostcall kernel is active.
Areas are created per context by [`hostcall_area`](@ref).
"""
mutable struct HostcallArea
    const ctx::CuContext
    const dev::CuDevice
    const nports::Int
    const mem::HostMemory
    const base::Ptr{UInt8}
    const layout::@NamedTuple{client::Int, inbox::Int, outbox::Int, header::Int,
                              packet::Int, total::Int}
    const locks::DeviceMemory
    const client::HostcallClientPtr
    const heartbeat::Bool               # poll while idle (exception-only kernels use this area)

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

function HostcallArea(ctx::CuContext, nports::Integer, exception_info::ExceptionInfo;
                      heartbeat::Bool=true)
    nports > 0 || throw(ArgumentError("the number of hostcall ports must be positive"))
    nports <= typemax(UInt32) - (HOSTCALL_SWEEP_CHUNK - 1) ||
        throw(ArgumentError("too many hostcall ports: $nports"))
    nports = cld(Int(nports), HOSTCALL_SWEEP_CHUNK) * HOSTCALL_SWEEP_CHUNK
    layout = hostcall_packet_layout(nports)
    # look up the device through the context: `context!` does not activate a context that
    # is already the task's current one, so `current_device()` may not work here yet
    dev = device(ctx)
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
        descriptor = HostcallClient(nports, dp(layout.inbox, UInt32),
                                    dp(layout.outbox, UInt32),
                                    dp(layout.header, HostcallHeader),
                                    dp(layout.packet, UInt8),
                                    reinterpret(LLVMPtr{UInt32,AS.Global},
                                                convert(CuPtr{UInt32}, locks)),
                                    reinterpret(Ptr{Cvoid}, exception_info))
        unsafe_store!(convert(Ptr{HostcallClient}, base + layout.client), descriptor)
        client = dp(layout.client, HostcallClient)
        shadow = Base.zeros(UInt32, nports)
        outbox = unsafe_wrap(Array, convert(Ptr{UInt32}, base + layout.outbox), nports)
        HostcallArea(ctx, dev, nports, mem, base, layout, locks, client, heartbeat,
                     shadow, outbox, 0, nothing)
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
    hostcall_area(ctx, exception_info; ports=HOSTCALL_MIN_PORTS) -> HostcallArea

Return the hostcall area of `ctx` with at least `ports` ports, creating it (and starting
the server thread) if necessary. Kernels that do not call host functions only need a small
area (for exception reporting), so contexts start with a minimal one and get a larger area
when a kernel that uses hostcalls is linked; earlier areas stay alive and keep being
serviced, because running kernels may still refer to them.
"""
function hostcall_area(ctx::CuContext, exception_info::ExceptionInfo;
                       ports::Integer=HOSTCALL_MIN_PORTS, heartbeat::Bool=true)
    @lock hostcall_areas_lock begin
        areas = Base.@atomic hostcall_areas.list
        # newest first: the last area created for a context is the largest
        for area in Iterators.reverse(areas)
            area.ctx == ctx && area.nports >= ports &&
                (!heartbeat || area.heartbeat) && return area
        end
        area = HostcallArea(ctx, ports, exception_info; heartbeat)
        hostcall_server_start()
        Base.@atomic hostcall_areas.list = [areas..., area]
        return area
    end
end

"""
    hostcall_client(ctx, exception_info, fallback; ports, heartbeat)

Pointer to a suitable runtime descriptor for kernels launched in `ctx`.
"""
function hostcall_client(ctx::CuContext, exception_info::ExceptionInfo,
                         fallback::HostcallClientPtr;
                         ports::Integer=HOSTCALL_MIN_PORTS, heartbeat::Bool=true)
    # kernels compiled during precompilation are not launched; creating the area would
    # start the server thread in the precompilation process. such kernels get the no-port
    # fallback descriptor, which is never dereferenced on the device.
    ccall(:jl_generating_output, Cint, ()) != 0 && return fallback
    return hostcall_area(ctx, exception_info; ports, heartbeat).client
end


## target registry

# a statically-known host function and how its arguments and result are encoded. `method`
# caches the MethodInstance that `jl_invoke` calls through, tagged with the world in which
# dispatch resolved it (see "invoking targets" below).
mutable struct HostcallTarget
    const key::Type          # registry key `Tuple{F,RT,AT}`
    const f::Any
    const RT::Type           # return type (`Nothing` for calls without a reply)
    const AT::Type           # payload tuple type; includes `f` when it is not stored
    const stored::Bool       # callable is stored in `f`, rather than included in the payload
    Base.@atomic method::Union{Nothing,Tuple{UInt,Core.MethodInstance}}
end

# targets keyed by the address of their rooted key type — the word the kernel's relocated
# literal slot holds. an immutable snapshot is swapped on registration, so the server
# thread reads it without locking (like `hostcall_areas`).
mutable struct HostcallTargets
    Base.@atomic table::Dict{UInt64,HostcallTarget}
end
const hostcall_targets = HostcallTargets(Dict{UInt64,HostcallTarget}())
const hostcall_targets_lock = Threads.SpinLock()

# The wire identifier of a key: the relocation resolver permanently roots and canonicalizes
# the key type, then returns its address. This matches the word in the image whether codegen
# baked the literal directly or `link_kernel` re-resolved a cached image's relocation slot.
hostcall_target_word(@nospecialize(K::Type)) =
    UInt64(GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(K)))

"""
    register_hostcall_targets!(keys)

Register the statically-known hostcall targets of a kernel (key types `Tuple{F,RT,AT}`, as
recorded by the compiler) so that the server can dispatch calls. Registration is
idempotent: a key's identifier is the address of the key type itself, so distinct keys
cannot collide.
"""
function register_hostcall_targets!(keys)
    isempty(keys) && return
    for K in keys
        id = hostcall_target_word(K)
        haskey(Base.@atomic(hostcall_targets.table), id) && continue
        F, RT, AT = K.parameters
        stored = Base.issingletontype(F)
        f = stored ? F.instance : nothing
        target = HostcallTarget(K, f, RT, AT, stored, nothing)

        # Compile the handler and resolve its MethodInstance on the registering thread,
        # before publishing the target: the server thread should not have to compile.
        seed_hostcall_method!(target)
        @lock hostcall_targets_lock begin
            table = Base.@atomic hostcall_targets.table
            if !haskey(table, id)
                table = copy(table)
                table[id] = target
                Base.@atomic hostcall_targets.table = table
            end
        end
    end
    return
end

hostcall_target(id::UInt64) = get(Base.@atomic(hostcall_targets.table), id, nothing)


## invoking targets

# The service path calls Julia's exported `jl_invoke`: resolve the handler's MethodInstance
# once per serviced call, then invoke every lane through Julia's own compiled fast path.
# The cache is tagged with the world counter because replacing a method does not invalidate
# the old MethodInstance: currency in the *latest* world — hostcalls behave like
# `invokelatest` — is a dispatch-level property that must be re-resolved when the world
# moves.

# the handler call signature, after the argument conversion `call_arguments` applies
hostconvert_host_type(@nospecialize(P)) = P <: LLVMPtr ? CuPtr{P.parameters[1]} : P
function hostcall_handler_sig(target::HostcallTarget)
    AT = target.AT
    params = Any[hostconvert_host_type(P) for P in AT.parameters]
    target.stored && pushfirst!(params, target.key.parameters[1])
    return Tuple{params...}
end

# Resolve and cache the handler's MethodInstance for the current world. This returns
# `nothing` when dispatch fails, in which case the service path uses `invokelatest` to
# report the ordinary MethodError. Resolution runs in the latest world explicitly: the
# server thread's adopted task remains in the world in which it was adopted.
function seed_hostcall_method!(target::HostcallTarget)
    world = Base.get_world_counter()
    method = Base.invoke_in_world(world, resolve_hostcall_method, target, world)
    Base.@atomic target.method = method
    return method
end
function resolve_hostcall_method(target::HostcallTarget, world::UInt)
    try
        sig = hostcall_handler_sig(target)
        precompile(sig)
        ft = sig.parameters[1]
        tt = Tuple{sig.parameters[2:end]...}
        mi = methodinstance(ft, tt, world)
        return (world, mi)
    catch
        return nothing
    end
end

# Call Julia's `jl_invoke`, which finds or compiles a CodeInstance for `mi` and invokes its
# boxed entry point. The handler is precompiled at registration, so the common path is the
# same atomic cache walk and indirect call Julia uses for an `invoke` expression. `args`
# excludes `f`, and the caller runs in the seed world (see `service_target!`).
@inline function invoke_hostcall_method(mi::Core.MethodInstance, @nospecialize(f),
                                        args::Vector{Any})
    GC.@preserve args begin
        ccall(:jl_invoke, Any, (Any, Ptr{Any}, UInt32, Any),
              f, pointer(args), length(args), mi)
    end
end

# conversion of values received from the device (pointers already arrive as `CuPtr`, see
# the device-side `hostconvert`)
hostconvert_host(@nospecialize(x)) = x
function hostconvert_host(@nospecialize(p::LLVMPtr))
    T = typeof(p).parameters[1]
    return load_bits(CuPtr{T}, reinterpret(Ptr{UInt8}, ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), p)))
end


## exceptions and deferred output

"""
    HostcallException

Thrown by `synchronize()` when a host function called from a kernel threw, when the
target of a call is unknown, or when its result could not be converted. The calling device
thread has been stopped. The original error and backtrace are available in the `error`
and `backtrace` fields, and the device the call came from in `device` (`nothing` for errors
in the server itself). Like device-side exceptions, these are reported by synchronizing the
context the kernel ran in.
"""
struct HostcallException <: Exception
    target::Any
    error::Any
    backtrace::Any
    device::Union{Nothing,CuDevice}
end
HostcallException(target, error, backtrace=nothing) =
    HostcallException(target, error, backtrace, nothing)

function Base.showerror(io::IO, err::HostcallException)
    print(io, "HostcallException: error while servicing a hostcall to ", err.target)
    err.device === nothing || print(io, " on device ", name(err.device))
    print(io, ":\n")
    showerror(io, err.error)
    if err.backtrace !== nothing
        println(io)
        Base.show_backtrace(io, err.backtrace)
    end
end

# pending exceptions, tagged with the context of the area they were recorded for (or
# `nothing` for errors in the server itself, which any context reports)
const hostcall_exceptions = Tuple{Union{Nothing,CuContext},HostcallException}[]
const hostcall_exceptions_lock = Threads.SpinLock()
const hostcall_exceptions_pending = Threads.Atomic{Int}(0)
# Aggregate work that must be observed at synchronization. Keeping the fast path to one
# atomic load avoids charging ordinary synchronization for each hostcall subsystem.
const hostcall_sync_pending = Threads.Atomic{Int}(0)

function record_hostcall_exception!(area::Union{Nothing,HostcallArea}, target, err, bt=nothing)
    ctx = area === nothing ? nothing : area.ctx
    dev = area === nothing ? nothing : area.dev
    @lock hostcall_exceptions_lock begin
        push!(hostcall_exceptions, (ctx, HostcallException(target, err, bt, dev)))
        Threads.atomic_add!(hostcall_exceptions_pending, 1)
        Threads.atomic_add!(hostcall_sync_pending, 1)
    end
    return
end

# throw the oldest pending exception of `ctx` (called from `check_exceptions`). exceptions
# are reported per context, like device-side exceptions: synchronizing one device does not
# surface the errors of kernels running on another.
function check_hostcall_exceptions(ctx::CuContext)
    hostcall_exceptions_pending[] == 0 && return
    err = @lock hostcall_exceptions_lock begin
        i = findfirst(((c, _),) -> c === nothing || c == ctx, hostcall_exceptions)
        i === nothing && return
        Threads.atomic_sub!(hostcall_exceptions_pending, 1)
        Threads.atomic_sub!(hostcall_sync_pending, 1)
        popat!(hostcall_exceptions, i)[2]
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
        Threads.atomic_add!(hostcall_output_pending, 1)
        Threads.atomic_add!(hostcall_sync_pending, 1)
    end
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
    data, n = @lock hostcall_output_lock begin
        n = hostcall_output_pending[]
        hostcall_output_pending[] = 0
        take!(hostcall_output), n
    end
    Threads.atomic_sub!(hostcall_sync_pending, n)
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

# everything below is deliberately type-erased (`@nospecialize`, `jl_new_bits`,
# `jl_value_ptr`): the server thread must not have to compile code for every new target
# type, only the handlers themselves, which are precompiled on registration.

# load a value of type `T` from memory into a new boxed object
@inline load_bits(@nospecialize(T::Type), ptr::Ptr{UInt8}) =
    ccall(:jl_new_bits, Any, (Any, Ptr{Cvoid}), T, ptr)

# store the bits of a boxed isbits value
@inline function store_bits!(ptr::Ptr{UInt8}, @nospecialize(x))
    n = Core.sizeof(typeof(x))
    n == 0 && return
    src = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), x)
    ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), ptr, src, n)
    return
end

# read every live lane's value of type `T`, receiving additional chunks if needed
function read_lanes(p::HostPort, @nospecialize(T::Type), mask::UInt32)
    values = Vector{Any}(undef, 32)
    nbytes = Core.sizeof(T)
    if nbytes <= HOSTCALL_PACKET_SIZE
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds values[lane + 1] = load_bits(T, lane_packet(p, lane))
        end
    else
        nchunks = cld(nbytes, HOSTCALL_PACKET_SIZE)
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
            @inbounds values[lane + 1] = load_bits(T, pointer(bufs[lane + 1]))
        end
    end
    return values
end

# write each lane's value (of type `T`) to its packets and hand the buffer back; for large
# values this involves multiple flips, for which the device must be receiving.
function write_lanes!(p::HostPort, @nospecialize(T::Type), values::Vector{Any}, mask::UInt32,
                      status::UInt32)
    nbytes = Core.sizeof(T)
    if nbytes <= HOSTCALL_PACKET_SIZE
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds store_bits!(lane_packet(p, lane), values[lane + 1])
        end
        hport_flip!(p, status)
    else
        nchunks = cld(nbytes, HOSTCALL_PACKET_SIZE)
        bufs = [Vector{UInt8}(undef, nchunks * HOSTCALL_PACKET_SIZE) for _ in 1:32]
        for lane in 0:31
            (mask >> lane) & 1 == 0 && continue
            @inbounds store_bits!(pointer(bufs[lane + 1]), values[lane + 1])
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

# the arguments of a call: the payload tuple without the callable (if shipped), converted
function call_arguments(@nospecialize(payload), skip::Int)
    n = nfields(payload) - skip
    args = Vector{Any}(undef, n)
    for i in 1:n
        @inbounds args[i] = hostconvert_host(getfield(payload, i + skip))
    end
    return args
end

# invoke a registered target for every live lane of a port
function service_target!(p::HostPort, target::HostcallTarget, hdr::HostcallHeader)
    # Resolve the handler's method once per call. A stale cache (the world moved since the
    # last resolution, e.g. the handler was redefined) is re-seeded before any lane calls.
    printing = target.stored && is_print_target(target.f)
    method = nothing
    if !printing
        method = Base.@atomic target.method
        if method === nothing || method[1] != Base.get_world_counter()
            method = seed_hostcall_method!(target)
        end
    end
    if method === nothing
        service_lanes!(p, target, hdr, nothing)
    else
        # Switch to the seed world once for the whole call, so `jl_invoke` and calls inside
        # the handler use the same latest-world snapshot.
        Base.invoke_in_world(method[1], service_lanes!, p, target, hdr, method[2])
    end
    return
end

function service_lanes!(p::HostPort, target::HostcallTarget, hdr::HostcallHeader,
                        mi::Union{Nothing,Core.MethodInstance})
    mask = hdr.mask
    RT = target.RT
    reply = (hdr.flags & HOSTCALL_FLAG_ASYNC) == 0 && RT !== Nothing
    printing = target.stored && is_print_target(target.f)
    payloads = read_lanes(p, target.AT, mask)
    results = Vector{Any}(undef, 32)
    status = HOSTCALL_STATUS_OK
    for lane in 0:31
        (mask >> lane) & 1 == 0 && continue
        payload = @inbounds payloads[lane + 1]
        f = target.stored ? target.f : getfield(payload, 1)
        args = call_arguments(payload, target.stored ? 0 : 1)
        try
            if printing
                queue_hostcall_output(f, args...)
                rv = nothing
            elseif mi !== nothing
                rv = invoke_hostcall_method(mi, f, args)
            else
                rv = Base.invokelatest(f, args...)
            end
            if reply
                rv isa RT || (rv = convert(RT, rv))
                @inbounds results[lane + 1] = rv
            end
        catch err
            record_hostcall_exception!(p.area, f, err, catch_backtrace())
            status = HOSTCALL_STATUS_ERROR
        end
    end
    if status != HOSTCALL_STATUS_OK || !reply
        hport_flip!(p, status)
    else
        write_lanes!(p, RT, results, mask, status)
    end
    return
end

# built-in targets, implemented by the runtime library and serviced by these handlers
const hostcall_builtins = Dict{UInt64,Function}(
    HC_EXCEPTION => service_exception_report,
    HC_OOM => service_oom_report,
)

function service_port!(a::HostcallArea, i::Int, out::UInt32)
    hdr = unsafe_load(header_ptr(a, i))
    p = HostPort(a, i, out, @inbounds a.shadow[i + 1])
    context!(a.ctx) do
        stream!(hostcall_stream(a)) do
            if hdr.target < HOSTCALL_BUILTIN_IDS
                handler = get(hostcall_builtins, hdr.target, nothing)
                if handler === nothing
                    record_hostcall_exception!(a, hdr.target, ErrorException("unknown built-in hostcall target"))
                    hport_flip!(p, HOSTCALL_STATUS_ERROR)
                else
                    try
                        handler(p, hdr)
                    catch err
                        record_hostcall_exception!(a, hdr.target, err, catch_backtrace())
                        hport_flip!(p, HOSTCALL_STATUS_ERROR)
                    end
                end
            else
                target = hostcall_target(hdr.target)
                if target === nothing
                    record_hostcall_exception!(a, hdr.target, ErrorException("unknown hostcall target; was the kernel compiled in another session without its targets being registered?"))
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
    # fast-path the all-idle case with a single scan over the whole area: the chunked walk
    # below has per-chunk overhead that adds up on slower CPUs, and with one server thread
    # sweeping every context's area, idle areas are the common case
    pending_ports(a) || return 0
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
    const serviced::Threads.Atomic{Int} # completed launches covered by a full sweep
    const graphs::Threads.Atomic{Bool}  # graph replays cannot be armed
    const sweep_owner::Threads.Atomic{UInt}  # identity of the task holding `sweep_lock`, or 0
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
    srv.sweep_owner[] = task_identity()
    return
end
function unlock_sweeps(srv::HostcallServer)
    srv.sweep_owner[] = UInt(0)
    @ccall uv_mutex_unlock(srv.sweep_lock::Ptr{Cvoid})::Cvoid
    return
end
# sweeps are owned by a task (the server thread's root task, or a draining task), not a
# thread: a draining task that migrates between threads keeps its ownership, and other
# tasks running on its thread do not inherit it.
task_identity() = UInt(pointer_from_objref(current_task()))
# whether the current task is in the middle of a sweep, i.e. running a handler
sweeping(srv::HostcallServer) = srv.sweep_owner[] == task_identity()
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

# Unarmed kernels can only use hostcalls for exception reporting. Poll just the areas
# assigned to those kernels; full-size areas are swept while their kernels are armed, or
# on every heartbeat once a hostcall graph has been captured.
function sweep_heartbeat!()
    n = 0
    for area in Base.@atomic hostcall_areas.list
        area.heartbeat || continue
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

# a short GC-safe sleep for the armed-but-idle path. Windows has no `usleep`, so use
# libuv's millisecond sleep there.
@static if Sys.iswindows()
    hostcall_backoff() = @gcsafe_ccall uv_sleep(1::Cuint)::Cvoid
else
    hostcall_backoff() = @gcsafe_ccall usleep(20::Cuint)::Cint
end

function hostcall_server_main(srv::HostcallServer)
    idle = 0
    while true
        try
            # account for completed launches
            while @ccall(uv_sem_trywait(srv.sem::Ptr{Cvoid})::Cint) == 0
                srv.finished += 1
            end
            armed = srv.launches[] - srv.finished
            full_sweep = armed > 0 || srv.finished != srv.serviced[] || srv.graphs[]

            lock_sweeps(srv)
            found = try
                full_sweep ? sweep_all!() : sweep_heartbeat!()
            finally
                unlock_sweeps(srv)
            end
            if full_sweep
                completed = srv.finished - srv.serviced[]
                srv.serviced[] = srv.finished
                completed > 0 && Threads.atomic_sub!(hostcall_sync_pending, completed)
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
                        hostcall_backoff()
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
            record_hostcall_exception!(nothing, :server, err, catch_backtrace())
        end
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
end

# NOTE: the adopted task's world age is fixed when the thread enters Julia and never
#       advances (the server never returns to top level), so anything the server dispatches
#       resolves in that world: later method definitions — handler redefinitions, but also
#       Revise edits to the server code itself — are only visible through `invokelatest` /
#       `invoke_in_world`, which is how handlers are resolved and invoked.
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
                             Threads.Atomic{Int}(0), Threads.Atomic{Bool}(false),
                             Threads.Atomic{UInt}(0), 0)

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

# Captured kernels are replayed without passing through `hostcall_launch`, so keep idle
# heartbeat sweeps enabled for every area after the first hostcall graph is captured.
function hostcall_mark_graph!()
    srv = hostcall_server[]::HostcallServer
    if !Threads.atomic_cas!(srv.graphs, false, true)
        Threads.atomic_add!(hostcall_sync_pending, 1)
    end
    return
end

"""
    hostcall_arm!()

Announce an upcoming launch of a kernel that may hostcall: the server starts polling.
Balanced by [`hostcall_disarm!`](@ref) after the kernel, enqueued on its stream.
"""
function hostcall_arm!()
    srv = hostcall_server[]::HostcallServer
    Threads.atomic_add!(hostcall_sync_pending, 1)
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

# Complete pending asynchronous work and surface handler failures. The aggregate counter
# keeps ordinary synchronization to one atomic load when no hostcall work is outstanding.
@inline function hostcall_synchronize(ctx::CuContext)
    hostcall_sync_pending[] == 0 && return
    hostcall_drain()
    check_hostcall_exceptions(ctx)
    return
end

# Forget the areas of a context that is about to be destroyed. Stop the server from
# sweeping them before releasing their streams and raw allocations.
function hostcall_forget!(pred)
    srv = hostcall_server[]
    srv === nothing && return
    nested = sweeping(srv)
    nested || lock_sweeps(srv)
    try
        @lock hostcall_areas_lock begin
            areas = Base.@atomic hostcall_areas.list
            forgotten = filter(pred, areas)
            isempty(forgotten) && return
            Base.@atomic hostcall_areas.list = filter(a -> !pred(a), areas)
            # Release resources while the context is still valid. These low-level memory
            # wrappers do not own finalizers.
            for a in forgotten
                context!(a.ctx) do
                    a.stream === nothing || unsafe_destroy!(a.stream)
                    a.stream = nothing
                    free(a.locks)
                    free(a.mem)
                end
            end
            # exceptions recorded for these contexts can no longer be reported
            ctxs = Set(a.ctx for a in forgotten)
            @lock hostcall_exceptions_lock begin
                n = length(hostcall_exceptions)
                filter!(((c, _),) -> !(c in ctxs), hostcall_exceptions)
                removed = n - length(hostcall_exceptions)
                Threads.atomic_sub!(hostcall_exceptions_pending, removed)
                Threads.atomic_sub!(hostcall_sync_pending, removed)
            end
        end
    finally
        nested || unlock_sweeps(srv)
    end
    return
end
hostcall_forget!(ctx::CuContext) = hostcall_forget!(a -> a.ctx == ctx)
hostcall_forget!(dev::CuDevice) = hostcall_forget!(a -> a.dev == dev)
