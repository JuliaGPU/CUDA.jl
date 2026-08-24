# Hostcall: calling host functions from device code.
#
# The protocol is a port of LLVM libc's GPU RPC ("mailbox"): a pinned, device-mapped host
# buffer holds, per port, an inbox word (written by the host), an outbox word (written by
# the device), a header, and one 64-byte packet per lane. Ownership of a port's buffer is
# encoded by the inbox and outbox bits: the device owns the port when they are equal, the
# host when they differ, and every send or receive is a single bit flip of the writer's own
# mailbox. Only loads, stores and fences touch the shared buffer (system-scope RMW atomics
# are not atomic across PCIe); intra-GPU arbitration uses a lock bitfield in device memory.
# Ports are claimed per warp: all lanes that reach a call site together share one port and
# each lane gets its own packet.

export @hostcall, hostcall, hostcall_async
@public HostcallClient, HostcallPort, HostcallHeader,
        hostcall_open, hostcall_send!, hostcall_recv!, hostcall_close!,
        hostcall_send_scalar!,
        hostcall_lane_packet, HOSTCALL_PACKET_SIZE, hostcall_packet_layout


## shared data structures

"""
    HostcallHeader

Per-port header written by the device when it opens a port: the mask of lanes that take
part in the call, a flag word, and the 64-bit target identifier.
"""
struct HostcallHeader
    mask::UInt32
    flags::UInt32
    target::UInt64
end

const HOSTCALL_PACKET_SIZE = 64      # bytes per lane
const HOSTCALL_LANES = 32            # packets per port
const HOSTCALL_PORT_BYTES = HOSTCALL_PACKET_SIZE * HOSTCALL_LANES

# Header flags used by the generic host service. Raw protocols may use the remaining bits.
const HOSTCALL_FLAG_ASYNC = UInt32(1)

# Built-in targets occupy the low identifiers; static targets have the high bit set.
const HOSTCALL_BUILTIN_IDS = UInt64(256)
const HOSTCALL_STATIC_ID_BIT = UInt64(0x8000_0000_0000_0000)
const HC_EXCEPTION = UInt64(1)
const HC_OOM = UInt64(2)

# inbox words carry a status in the bits above the ownership bit; the host sets these
# when it replies to a port. bit 0 is the ownership bit.
const HOSTCALL_STATUS_OK = UInt32(0)
const HOSTCALL_STATUS_ERROR = UInt32(1)    # the handler threw, or the target is unknown

"""
    HostcallClient

Device-side descriptor of a hostcall area: the number of ports and pointers to the
mailboxes, headers, packets (all in pinned host memory) and the lock bitfield (in device
memory). A client with `nports == 0` is a placeholder used for kernels compiled during
precompilation, which are never launched.
The descriptor also points to the context's exception state. The kernel state holds only
a pointer to this descriptor, keeping it small (it is passed to child launches during
dynamic parallelism).
"""
struct HostcallClient
    nports::UInt32
    inbox::LLVMPtr{UInt32,AS.Global}            # host-written, device-read
    outbox::LLVMPtr{UInt32,AS.Global}           # device-written, host-read
    header::LLVMPtr{HostcallHeader,AS.Global}
    packet::LLVMPtr{UInt8,AS.Global}            # nports * 32 lanes * 64 bytes
    lock::LLVMPtr{UInt32,AS.Global}             # device memory, nports/32 words
    exception_info::Ptr{Cvoid}
end

HostcallClient(exception_info::Ptr{Cvoid}=C_NULL) =
    HostcallClient(0, reinterpret(LLVMPtr{UInt32,AS.Global}, C_NULL),
                   reinterpret(LLVMPtr{UInt32,AS.Global}, C_NULL),
                   reinterpret(LLVMPtr{HostcallHeader,AS.Global}, C_NULL),
                   reinterpret(LLVMPtr{UInt8,AS.Global}, C_NULL),
                   reinterpret(LLVMPtr{UInt32,AS.Global}, C_NULL), exception_info)

# protocol-only constructor, for raw-port users that do not carry exception state
HostcallClient(nports, inbox, outbox, header, packet, lock) =
    HostcallClient(nports, inbox, outbox, header, packet, lock, C_NULL)

const HostcallClientPtr = LLVMPtr{HostcallClient,AS.Global}
const null_hostcall_client = reinterpret(HostcallClientPtr, C_NULL)

"""
    hostcall_packet_layout(nports)

Compute the byte layout of a hostcall area with `nports` ports. Returns a named tuple with
the offsets of the client descriptor, inbox, outbox, header and packet arrays, and the
total size. The layout is shared between the device-side client and the host-side area.
"""
function hostcall_packet_layout(nports::Integer)
    nports >= 0 || throw(ArgumentError("the number of hostcall ports must be non-negative"))
    nports <= typemax(UInt32) || throw(ArgumentError("too many hostcall ports: $nports"))
    nports = Int(nports)
    client = 0
    # Keep the immutable descriptor off the cache line containing the first mailboxes.
    inbox = Base.checked_mul(cld(sizeof(HostcallClient), 128), 128)
    outbox = Base.checked_add(inbox, Base.checked_mul(4, nports))
    header = Base.checked_add(outbox, Base.checked_mul(4, nports))
    packet = Base.checked_mul(cld(Base.checked_add(header,
                                                   Base.checked_mul(sizeof(HostcallHeader), nports)),
                                  128), 128)
    total = Base.checked_add(packet, Base.checked_mul(nports, HOSTCALL_PORT_BYTES))
    return (; client, inbox, outbox, header, packet, total)
end

"""
    HostcallPort

A claimed hostcall port: the client it belongs to, the mask of participating lanes, the
port index, the current value of the outbox bit and whether the device currently owns the
buffer. Ports are immutable values; [`hostcall_send!`](@ref) and [`hostcall_recv!`](@ref)
return the updated port.
"""
struct HostcallPort
    client::HostcallClient
    mask::UInt32
    index::UInt32
    out::UInt32
    owns::Bool
end


## memory model primitives

# mailbox accesses are relaxed system-scope loads/stores (sm_70+), or volatile accesses on
# older hardware, which is what LLVM libc does as well. inline assembly keeps LLVM from
# hoisting or combining them.
@inline function mailbox_load(p::LLVMPtr{UInt32,AS.Global})
    if compute_capability() >= sv"7.0"
        @asmcall("ld.relaxed.sys.global.u32 \$0, [\$1];", "=r,l", true,
                 UInt32, Tuple{LLVMPtr{UInt32,AS.Global}}, p)
    else
        @asmcall("ld.volatile.global.u32 \$0, [\$1];", "=r,l", true,
                 UInt32, Tuple{LLVMPtr{UInt32,AS.Global}}, p)
    end
end

@inline function mailbox_store!(p::LLVMPtr{UInt32,AS.Global}, v::UInt32)
    if compute_capability() >= sv"7.0"
        @asmcall("st.relaxed.sys.global.u32 [\$0], \$1;", "l,r", true,
                 Cvoid, Tuple{LLVMPtr{UInt32,AS.Global},UInt32}, p, v)
    else
        @asmcall("st.volatile.global.u32 [\$0], \$1;", "l,r", true,
                 Cvoid, Tuple{LLVMPtr{UInt32,AS.Global},UInt32}, p, v)
    end
end

# system-scope fence (the PTX fence is acquire-release; before sm_70 use membar.sys)
@inline function fence_sys()
    if compute_capability() >= sv"7.0"
        @asmcall("fence.acq_rel.sys;", "~{memory}", true, Cvoid, Tuple{})
    else
        threadfence_system()
    end
end

# gpu-scope fence, for the lock bitfield in device memory
@inline function fence_gpu()
    if compute_capability() >= sv"7.0"
        @asmcall("fence.acq_rel.gpu;", "~{memory}", true, Cvoid, Tuple{})
    else
        threadfence()
    end
end

# exponential backoff between polls; sleeping is not available before sm_70, in which case
# we simply re-probe (the load itself takes a few hundred nanoseconds).
const HOSTCALL_BACKOFF_MIN = UInt32(8)
const HOSTCALL_BACKOFF_MAX = UInt32(256)
@inline function hostcall_backoff(ns::UInt32)
    if compute_capability() >= sv"7.0"
        # Do not call `nanosleep` here: its static capability check runs before this
        # target-dependent branch is eliminated on some Julia versions.
        @asmcall("nanosleep.u32 \$0;", "r", true, Cvoid, Tuple{UInt32}, ns)
    end
    return min(ns << 1, HOSTCALL_BACKOFF_MAX)
end


## warp helpers

@inline first_lane(mask::UInt32) = trailing_zeros(mask) % Int32 + Int32(1)   # 1-based
@inline is_first_lane(mask::UInt32) = laneid() == first_lane(mask)
@inline broadcast_value(mask::UInt32, v) = shfl_sync(mask, v, first_lane(mask))

# spread concurrently active warps over the port space; starting every warp at port 0
# heavily contends the first ports.
@inline function hostcall_start_index(nports::UInt32)
    block = (blockIdx().z - 1i32) * gridDim().y * gridDim().x +
            (blockIdx().y - 1i32) * gridDim().x + (blockIdx().x - 1i32)
    thread = (threadIdx().z - 1i32) * blockDim().y * blockDim().x +
             (threadIdx().y - 1i32) * blockDim().x + (threadIdx().x - 1i32)
    warps_per_block = (blockDim().x * blockDim().y * blockDim().z + 31i32) ÷ 32i32
    warp = block * warps_per_block + thread ÷ 32i32
    return (warp % UInt32) % nports
end


## lock bitfield (device memory)

@inline function try_lock!(c::HostcallClient, mask::UInt32, index::UInt32)
    # every lane in the mask atomically sets the port's bit; the ballot tells whether any
    # lane observed it clear, i.e. whether the warp took the lock. lanes outside the mask
    # (under independent thread scheduling) are no-ops.
    id = laneid() - Int32(1)
    in_mask = (mask >> id) & UInt32(1)
    word = c.lock + 4 * (index >> 5)
    bit = UInt32(1) << (index & UInt32(31))
    before = atomic_or!(word, in_mask * bit)
    failed = (before & bit) != 0
    packed = vote_ballot_sync(mask, failed)
    holding = mask != packed
    holding && fence_gpu()
    return holding
end

@inline function unlock!(c::HostcallClient, mask::UInt32, index::UInt32)
    fence_gpu()
    if is_first_lane(mask)
        word = c.lock + 4 * (index >> 5)
        bit = UInt32(1) << (index & UInt32(31))
        atomic_and!(word, ~bit)
    end
    sync_warp(mask)
    return
end


## port protocol

@inline hostcall_load_inbox(c::HostcallClient, mask::UInt32, index::UInt32) =
    broadcast_value(mask, mailbox_load(c.inbox + 4index))

@inline hostcall_load_outbox(c::HostcallClient, mask::UInt32, index::UInt32) =
    broadcast_value(mask, mailbox_load(c.outbox + 4index))

# whether the device owns the buffer, given the inbox and outbox words (bit 0 only; the
# remaining inbox bits carry a status)
@inline hostcall_owned(in::UInt32, out::UInt32) = (in & UInt32(1)) == out

@inline function hostcall_invert_outbox!(c::HostcallClient, mask::UInt32, index::UInt32,
                                         out::UInt32)
    inverted = out ⊻ UInt32(1)
    sync_warp(mask)
    fence_sys()
    if is_first_lane(mask)
        mailbox_store!(c.outbox + 4index, inverted)
    end
    return inverted
end

# spin until the inbox indicates that the device owns the buffer; returns the inbox word
@inline function hostcall_wait_for_ownership(c::HostcallClient, mask::UInt32, index::UInt32,
                                             out::UInt32, in::UInt32)
    ns = HOSTCALL_BACKOFF_MIN
    while !hostcall_owned(in, out)
        ns = hostcall_backoff(ns)
        in = hostcall_load_inbox(c, mask, index)
    end
    fence_sys()
    return in
end

"""
    hostcall_open(client::HostcallClient, target::UInt64, flags::UInt32=0) -> HostcallPort

Claim a free port and label it with `target`. This is a warp-collective operation: all
lanes that are active when calling it share the port, and the active mask is captured for
the lifetime of the port. The device owns the buffer of the returned port.
"""
@inline function hostcall_open(c::HostcallClient, target::UInt64, flags::UInt32=UInt32(0))
    index = hostcall_start_index(c.nports)
    while true
        # under independent thread scheduling the lanes may reconverge with different
        # indices, so re-read the mask and keep the index uniform
        mask = active_mask()
        index = broadcast_value(mask, index)
        if try_lock!(c, mask, index)
            # issue both loads before broadcasting either; they are independent
            in_raw = mailbox_load(c.inbox + 4index)
            out_raw = mailbox_load(c.outbox + 4index)
            in = broadcast_value(mask, in_raw)
            out = broadcast_value(mask, out_raw)
            if hostcall_owned(in, out)
                if is_first_lane(mask)
                    unsafe_store!(c.header + sizeof(HostcallHeader) * index,
                                  HostcallHeader(mask, flags, target))
                end
                sync_warp(mask)
                return HostcallPort(c, mask, index, out, true)
            end
            # the port is free but its last call has not been serviced yet
            unlock!(c, mask, index)
        end
        index += UInt32(1)
        index >= c.nports && (index = UInt32(0))
    end
end

"""
    hostcall_lane_packet(port::HostcallPort) -> LLVMPtr{UInt8,AS.Global}

Pointer to the calling lane's 64-byte packet of `port`.
"""
@inline hostcall_lane_packet(port::HostcallPort) =
    port.client.packet + (port.index * HOSTCALL_LANES + (laneid() - 1i32)) * HOSTCALL_PACKET_SIZE

"""
    hostcall_send!(fill, port::HostcallPort) -> HostcallPort

Wait until the device owns the buffer of `port`, call `fill(packet)` on every lane with a
pointer to the lane's packet, and hand the buffer to the host. Returns the updated port.

Note that `fill` runs on the device; variables it captures must not be reassigned in the
enclosing function (use `let`), or they get boxed.
"""
@inline function hostcall_send!(fill::F, port::HostcallPort) where {F}
    c = port.client
    in = port.owns ? port.out : hostcall_load_inbox(c, port.mask, port.index)
    hostcall_wait_for_ownership(c, port.mask, port.index, port.out, in)
    fill(hostcall_lane_packet(port))
    out = hostcall_invert_outbox!(c, port.mask, port.index, port.out)
    return HostcallPort(c, port.mask, port.index, out, false)
end

"""
    hostcall_recv!(use, port::HostcallPort) -> (HostcallPort, value, status)

Wait until the host has handed the buffer of `port` back, and call `use(packet)` on every
lane with a pointer to the lane's packet. Returns the updated port, the value returned by
`use`, and the status word set by the host (`0` on success).
"""
@inline function hostcall_recv!(use::U, port::HostcallPort) where {U}
    c = port.client
    out = port.out
    if port.owns
        # consecutive receives: hand the buffer back first
        out = hostcall_invert_outbox!(c, port.mask, port.index, out)
    end
    in = hostcall_load_inbox(c, port.mask, port.index)
    in = hostcall_wait_for_ownership(c, port.mask, port.index, out, in)
    val = use(hostcall_lane_packet(port))
    return HostcallPort(c, port.mask, port.index, out, true), val, in >> 1
end

"""
    hostcall_close!(port::HostcallPort)

Release `port`. If the host still owns the buffer (i.e. after a send without a matching
receive) the call is completed asynchronously by the host.
"""
@inline function hostcall_close!(port::HostcallPort)
    sync_warp(port.mask)
    unlock!(port.client, port.mask, port.index)
    return
end

"""
    hostcall_send_scalar!(client::HostcallClient, target::UInt64, value)

Send a single `value` (at most one packet in size) to `target` from the calling lane,
without waiting for the host to service the call. Unlike the warp-collective port API,
this may be called from arbitrarily divergent code, on any hardware: it involves no warp
intrinsics. The runtime uses it for exception and out-of-memory reports.
"""
# Keep this inline: CUDA 12.9 ptxas crashes on the debug information for an out-of-line
# function taking an aggregate value by value. Inlining also produces less PTX.
@inline function hostcall_send_scalar!(c::HostcallClient, target::UInt64,
                                       value::T) where {T}
    GPUCompiler.@static_assert(sizeof(T) <= HOSTCALL_PACKET_SIZE,
                               "hostcall_send_scalar! values must fit in one packet")
    index = hostcall_start_index(c.nports)
    lane = laneid() - Int32(1)
    mask = UInt32(1) << lane
    while true
        word = c.lock + 4 * (index >> 5)
        bit = UInt32(1) << (index & UInt32(31))
        if (atomic_or!(word, bit) & bit) == 0
            fence_gpu()
            in = mailbox_load(c.inbox + 4index)
            out = mailbox_load(c.outbox + 4index)
            if hostcall_owned(in, out)
                unsafe_store!(c.header + sizeof(HostcallHeader) * index,
                              HostcallHeader(mask, UInt32(0), target))
                packet = c.packet + (index * HOSTCALL_LANES + lane) * HOSTCALL_PACKET_SIZE
                unsafe_store!(reinterpret(LLVMPtr{T,AS.Global}, packet), value)
                fence_sys()
                mailbox_store!(c.outbox + 4index, out ⊻ UInt32(1))
                fence_gpu()
                atomic_and!(word, ~bit)
                return
            end
            fence_gpu()
            atomic_and!(word, ~bit)
        end
        index += UInt32(1)
        index >= c.nports && (index = UInt32(0))
    end
end


## value marshalling

# values are shipped in their Julia layout, 64 bytes per packet; larger values are split
# over consecutive sends (the host knows the type, and thus the number of chunks)

@inline function hostcall_send_value!(port::HostcallPort, x::T) where {T}
    if sizeof(T) <= HOSTCALL_PACKET_SIZE
        port = hostcall_send!(port) do pkt
            unsafe_store!(reinterpret(LLVMPtr{T,AS.Global}, pkt), x)
        end
    else
        # stream the value through local memory; all lanes send the same number of chunks
        ref = Ref(x)
        GC.@preserve ref begin
            src = reinterpret(LLVMPtr{UInt8,AS.Generic}, Base.unsafe_convert(Ptr{T}, ref))
            nchunks = cld(sizeof(T), HOSTCALL_PACKET_SIZE)
            i = 0
            while i < nchunks
                offset = i * HOSTCALL_PACKET_SIZE
                nbytes = min(HOSTCALL_PACKET_SIZE, sizeof(T) - offset)
                port = let offset = offset, nbytes = nbytes, src = src
                    hostcall_send!(port) do pkt
                        j = 0
                        while j < nbytes
                            unsafe_store!(pkt + j, unsafe_load(src + offset + j))
                            j += 1
                        end
                    end
                end
                i += 1
            end
        end
    end
    return port
end

@inline function hostcall_recv_value!(port::HostcallPort, ::Type{T}) where {T}
    if sizeof(T) <= HOSTCALL_PACKET_SIZE
        port, val, status = hostcall_recv!(port) do pkt
            unsafe_load(reinterpret(LLVMPtr{T,AS.Global}, pkt))
        end
        return port, val, status
    else
        ref = Ref{T}()
        status = UInt32(0)
        GC.@preserve ref begin
            dst = reinterpret(LLVMPtr{UInt8,AS.Generic}, Base.unsafe_convert(Ptr{T}, ref))
            nchunks = cld(sizeof(T), HOSTCALL_PACKET_SIZE)
            i = 0
            while i < nchunks
                offset = i * HOSTCALL_PACKET_SIZE
                nbytes = min(HOSTCALL_PACKET_SIZE, sizeof(T) - offset)
                port, _, st = let offset = offset, nbytes = nbytes, dst = dst
                    hostcall_recv!(port) do pkt
                        j = 0
                        while j < nbytes
                            unsafe_store!(dst + offset + j, unsafe_load(pkt + j))
                            j += 1
                        end
                        nothing
                    end
                end
                status |= st
                # Error replies contain no result payload. In particular, the host may not
                # know the result type of an unknown target and can only return one packet.
                st == HOSTCALL_STATUS_OK || break
                i += 1
            end
            return port, ref[], status
        end
    end
end


## high-level API

# Device-side argument conversion. Arguments travel in their Julia layout and may contain
# compiler-relocated host constants such as string literals. Device pointers are shipped as
# `CuPtr`, which is what the host expects.
hostconvert(x) = x
hostconvert(p::LLVMPtr{T}) where {T} = reinterpret(CuPtr{T}, p)

# The hash is computed while compiling and travels with the image. The high bit keeps
# static targets disjoint from built-ins; the registry detects collisions.
function hostcall_target_id_value(@nospecialize(K::Type))
    return (hash(K) % UInt64) | HOSTCALL_STATIC_ID_BIT
end
@generated hostcall_target_id(::Type{K}) where {K} = :($(hostcall_target_id_value(K)))

# the marker function the compiler scans for: `K` is the registry key
# (`Tuple{typeof(f), RT, AT}`). it must not be inlined so that its specializations show up
# in the compiled method instances, and must not throw.
@noinline function hostcall_impl(::Type{K}, ::Type{RT}, args::AT,
                                 ::Val{async}) where {K, RT, AT<:Tuple, async}
    client = hostcall_client()
    flags = async ? HOSTCALL_FLAG_ASYNC : UInt32(0)
    port = hostcall_open(client, hostcall_target_id(K), flags)
    port = hostcall_send_value!(port, args)
    if async
        hostcall_close!(port)
        return nothing
    end
    port, val, status = hostcall_recv_value!(port, RT)
    hostcall_close!(port)
    if status != HOSTCALL_STATUS_OK
        # the handler failed; the host has recorded the error and will throw it at the next
        # synchronization, so just stop this thread.
        exit()
    end
    return val
end

@inline hostcall_key(::F, ::Type{RT}, ::Type{AT}) where {F,RT,AT} = Tuple{F,RT,AT}

# the values shipped for a call: the function itself unless it is a singleton
@inline hostcall_payload(f::F, args::Tuple) where {F} =
    Base.issingletontype(F) ? args : (f, args...)

"""
    hostcall(f, R, args...) -> R
    hostcall_async(f, args...) -> nothing

Call the host function `f` with `args...` from device code, returning its result converted
to `R`. `hostcall_async` does not wait for the call to complete and implies `R === Nothing`.
`R` must be isbits or `Nothing`. Arguments may contain compiler-relocated host constants,
such as string literals, but arbitrary Julia references are unsupported. `f` must be
recoverable from its type (a named function or an isbits functor). See [`@hostcall`](@ref).
"""
@inline function hostcall(f::F, ::Type{RT}, args...) where {F,RT}
    # Results are reconstructed on the device and cannot contain host references. Argument
    # payloads may contain compiler-relocated constants such as string literals.
    GPUCompiler.@static_assert(RT === Nothing || isbitstype(RT),
                               "hostcall return types must be isbits or Nothing")
    payload = hostcall_payload(f, map(hostconvert, args))
    K = hostcall_key(f, RT, typeof(payload))
    return hostcall_impl(K, RT, payload, Val(false))::RT
end

@inline function hostcall_async(f::F, args...) where {F}
    payload = hostcall_payload(f, map(hostconvert, args))
    K = hostcall_key(f, Nothing, typeof(payload))
    hostcall_impl(K, Nothing, payload, Val(true))
    return nothing
end

@doc (@doc hostcall) hostcall_async

"""
    @hostcall f(args...)::R
    @hostcall async=true f(args...)

Call the host function `f` from device code, `@ccall`-style: the return type annotation
`::R` is required (it is the one thing the device cannot know) unless `async=true`, in
which case the call returns immediately, `R` is `Nothing`, and the call is completed by the
time the next `synchronize()` returns. Individual arguments may be annotated (`a::T`) to
convert them before shipping.

The call is warp-collective: all active lanes submit their own arguments and receive their
own results through a single port; divergent lanes simply form separate calls. Values are
shipped in their Julia layout. Arguments may contain compiler-relocated host constants such
as string literals, but arbitrary Julia references are unsupported; results must be isbits
or `Nothing`. Device memory is passed as explicit pointers (received as `CuPtr` on the host).
The handler runs on a dedicated host thread with the kernel's context active and a dedicated
non-blocking stream; handler exceptions surface as `HostcallException` at the next
`synchronize()`.

```julia
function kernel(out, i)
    y = @hostcall load_from_disk(i)::Float32
    @hostcall async=true println("thread ", i)
    ...
end
```
"""
macro hostcall(exprs...)
    isempty(exprs) && throw(ArgumentError("@hostcall requires a call expression"))
    call = exprs[end]
    options = exprs[1:end-1]

    # options
    async = false
    for opt in options
        Meta.isexpr(opt, :(=), 2) ||
            throw(ArgumentError("invalid @hostcall option `$opt`; expected `key=value`"))
        key, val = opt.args
        if key === :async
            val isa Bool ||
                throw(ArgumentError("the `async` option of @hostcall requires a literal Bool"))
            async = val
        else
            throw(ArgumentError("unknown @hostcall option `$key`"))
        end
    end

    # return type
    rettype = nothing
    if Meta.isexpr(call, :(::), 2)
        rettype = call.args[2]
        call = call.args[1]
    end
    if !async && rettype === nothing
        throw(ArgumentError("@hostcall requires a return type annotation: `@hostcall f(args...)::R`, or `async=true`"))
    end
    if async && rettype !== nothing && rettype !== :Nothing
        throw(ArgumentError("`@hostcall async=true` cannot return a value; drop the `::$rettype` annotation"))
    end
    Meta.isexpr(call, :call) ||
        throw(ArgumentError("@hostcall expects a function call, got `$call`"))
    f = call.args[1]
    args = call.args[2:end]
    any(arg -> Meta.isexpr(arg, :parameters) || Meta.isexpr(arg, :kw), args) &&
        throw(ArgumentError("@hostcall does not support keyword arguments"))

    # per-argument conversions
    argexprs = map(args) do arg
        if Meta.isexpr(arg, :(::), 2)
            :(convert($(esc(arg.args[2])), $(esc(arg.args[1]))))
        else
            esc(arg)
        end
    end

    if async
        :(hostcall_async($(esc(f)), $(argexprs...)))
    else
        :(hostcall($(esc(f)), $(esc(rettype)), $(argexprs...)))
    end
end
