using CUDA: HostcallClient, HostcallPort, HostcallHeader,
            hostcall_open, hostcall_send!, hostcall_recv!, hostcall_close!,
            hostcall_lane_packet, HOSTCALL_PACKET_SIZE, hostcall_packet_layout

@testset "@hostcall syntax" begin
    @test_throws ArgumentError hostcall_packet_layout(-1)
    f(x) = x
    # the return type is required unless async
    @test_throws LoadError @eval @hostcall f(1)
    @test_throws LoadError @eval @hostcall async=true f(1)::Int
    @test_throws LoadError @eval @hostcall lanes=:all f(1)::Int
    @test_throws LoadError @eval @hostcall f(1; y=2)::Int
    ex = @macroexpand @hostcall f(1, 2.0)::Int
    @test Meta.isexpr(ex, :call) && ex.args[1] == GlobalRef(CUDACore, :hostcall)
    @test ex.args[3] == :Int
    ex = @macroexpand @hostcall async=true f(1)
    @test Meta.isexpr(ex, :call) && ex.args[1] == GlobalRef(CUDACore, :hostcall_async)
    ex = @macroexpand @hostcall async=true f(1)::Nothing
    @test Meta.isexpr(ex, :call) && ex.args[1] == GlobalRef(CUDACore, :hostcall_async)
    ex = @macroexpand @hostcall f(x::Float32)::Int
    @test Meta.isexpr(ex.args[4], :call) && ex.args[4].args[1].name == :convert

    # non-isbits return types are rejected when the kernel is compiled
    badrt() = (@hostcall string(1)::String; nothing)
    @test_throws "hostcall return types must be isbits" @cuda launch=false badrt()
end

# a minimal host-side implementation of the port protocol, so that the device-side API can
# be tested without the hostcall service: the test polls the area on the main thread.
module TestPoller
    using CUDA, CUDACore
    using CUDA: HostcallClient, HostcallHeader, HOSTCALL_PACKET_SIZE, hostcall_packet_layout
    using Core: LLVMPtr
    const AS = CUDA.AS

    mutable struct Area
        nports::Int
        mem::CUDACore.HostMemory
        base::Ptr{UInt8}
        locks::CuVector{UInt32}
        client::HostcallClient
        layout::Any
    end

    function Area(nports)
        layout = hostcall_packet_layout(nports)
        mem = CUDACore.alloc(CUDACore.HostMemory, layout.total,
                             CUDACore.MEMHOSTALLOC_DEVICEMAP | CUDACore.MEMHOSTALLOC_PORTABLE)
        base = convert(Ptr{UInt8}, mem)
        @assert UInt(base) == UInt(convert(CuPtr{UInt8}, mem))
        unsafe_wrap(Array, base, layout.total) .= 0
        locks = CUDA.zeros(UInt32, cld(nports, 32))
        dp(off, T) = reinterpret(LLVMPtr{T,AS.Global}, base + off)
        client = HostcallClient(nports, dp(layout.inbox, UInt32), dp(layout.outbox, UInt32),
                                dp(layout.header, HostcallHeader), dp(layout.packet, UInt8),
                                reinterpret(LLVMPtr{UInt32,AS.Global}, pointer(locks)))
        Area(nports, mem, base, locks, client, layout)
    end
    free(a::Area) = CUDACore.free(a.mem)

    inbox(a, i) = convert(Ptr{UInt32}, a.base + a.layout.inbox) + 4i
    outbox(a, i) = convert(Ptr{UInt32}, a.base + a.layout.outbox) + 4i
    header(a, i) = unsafe_load(convert(Ptr{HostcallHeader}, a.base + a.layout.header) + sizeof(HostcallHeader) * i)
    packet(a, i, lane) = a.base + a.layout.packet + (i * 32 + lane) * HOSTCALL_PACKET_SIZE

    load(p::Ptr{UInt32}) = Core.Intrinsics.atomic_pointerref(p, :acquire)
    store!(p::Ptr{UInt32}, v::UInt32) = Core.Intrinsics.atomic_pointerset(p, v, :release)

    # one sweep; `handle(target, lane, packet_ptr)` services every live lane of a pending
    # port and returns the status to report. returns the number of ports serviced.
    function sweep!(handle, a::Area)
        n = 0
        for i in 0:a.nports-1
            out = load(outbox(a, i))
            in = load(inbox(a, i))
            (in & 1) == out && continue
            hdr = header(a, i)
            status = UInt32(0)
            for lane in 0:31
                (hdr.mask >> lane) & 1 == 0 && continue
                status |= UInt32(handle(hdr.target, lane, packet(a, i, lane)))
            end
            store!(inbox(a, i), out | (status << 1))
            n += 1
        end
        return n
    end

    # poll until the current stream is done and everything has been drained
    function serve!(handle, a::Area; timeout=30)
        t0 = time()
        while true
            n = sweep!(handle, a)
            if n == 0
                CUDA.isdone(stream()) && break
                ccall(:jl_cpu_pause, Cvoid, ())
                time() - t0 > timeout && error("timeout waiting for the kernel")
            end
        end
        sweep!(handle, a)
        return
    end
end

@testset "raw ports" begin
    area = TestPoller.Area(64)
    try
        OP_ADD1 = UInt64(1)
        OP_SUM4 = UInt64(2)
        OP_RECORD = UInt64(3)
        OP_FAIL = UInt64(4)
        records = Threads.Atomic{Int}(0)
        function handler(target, lane, pkt)
            p = convert(Ptr{UInt64}, pkt)
            if target == OP_ADD1
                unsafe_store!(p, unsafe_load(p) + 1)
            elseif target == OP_SUM4
                unsafe_store!(p, sum(unsafe_load(p, i) for i in 1:4))
            elseif target == OP_RECORD
                Threads.atomic_add!(records, 1)
            elseif target == OP_FAIL
                return 1
            end
            return 0
        end

        # blocking round trips, every lane its own value
        function roundtrip(client, n, out)
            v = UInt64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
            i = 0
            while i < n
                port = hostcall_open(client, OP_ADD1)
                port = let v = v
                    hostcall_send!(port) do pkt
                        unsafe_store!(reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt), v)
                    end
                end
                port, v, status = hostcall_recv!(port) do pkt
                    unsafe_load(reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt))
                end
                hostcall_close!(port)
                i += 1
            end
            out[(blockIdx().x - 1) * blockDim().x + threadIdx().x] = v
            return
        end
        for (threads, blocks) in [(32, 1), (20, 2), (256, 4)]
            out = CUDA.zeros(UInt64, threads * blocks)
            @cuda threads=threads blocks=blocks roundtrip(area.client, 3, out)
            TestPoller.serve!(handler, area)
            synchronize()
            @test Array(out) == UInt64.(1:threads*blocks) .+ 3
        end

        # divergent call sites within a warp
        function divergent(client, out)
            t = threadIdx().x
            if isodd(t)
                port = hostcall_open(client, OP_SUM4)
                port = hostcall_send!(port) do pkt
                    p = reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt)
                    unsafe_store!(p, UInt64(t), 1)
                    unsafe_store!(p, UInt64(10), 2)
                    unsafe_store!(p, UInt64(100), 3)
                    unsafe_store!(p, UInt64(1000), 4)
                end
                port, r, _ = hostcall_recv!(port) do pkt
                    unsafe_load(reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt))
                end
                hostcall_close!(port)
                out[t] = r
            else
                port = hostcall_open(client, OP_ADD1)
                port = hostcall_send!(port) do pkt
                    unsafe_store!(reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt), UInt64(t))
                end
                port, r, _ = hostcall_recv!(port) do pkt
                    unsafe_load(reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt))
                end
                hostcall_close!(port)
                out[t] = r
            end
            return
        end
        out = CUDA.zeros(UInt64, 32)
        @cuda threads=32 divergent(area.client, out)
        TestPoller.serve!(handler, area)
        synchronize()
        @test Array(out) == [isodd(t) ? UInt64(t + 1110) : UInt64(t + 1) for t in 1:32]

        # asynchronous sends: the port is released without waiting for a reply
        function record(client, n)
            i = 0
            while i < n
                port = hostcall_open(client, OP_RECORD)
                port = let i = i
                    hostcall_send!(port) do pkt
                        unsafe_store!(reinterpret(Core.LLVMPtr{Int,AS.Global}, pkt), i)
                    end
                end
                hostcall_close!(port)
                i += 1
            end
            return
        end
        records[] = 0
        @cuda threads=64 blocks=8 record(area.client, 5)
        TestPoller.serve!(handler, area)
        synchronize()
        @test records[] == 64 * 8 * 5

        # a status word set by the host is returned by recv!
        function failing(client, out)
            port = hostcall_open(client, OP_FAIL)
            port = hostcall_send!(port) do pkt end
            port, _, status = hostcall_recv!(port) do pkt nothing end
            hostcall_close!(port)
            out[threadIdx().x] = status
            return
        end
        out = CUDA.zeros(UInt32, 4)
        @cuda threads=4 failing(area.client, out)
        TestPoller.serve!(handler, area)
        synchronize()
        @test all(==(1), Array(out))

        # all ports are released afterwards
        @test all(==(0), Array(area.locks))
    finally
        TestPoller.free(area)
    end
end

@testset "memory model" begin
    # the mailbox accesses should be system-scope, the fences and sleeps present
    function probe(client, out)
        port = hostcall_open(client, UInt64(1))
        port = hostcall_send!(port) do pkt end
        port, v, _ = hostcall_recv!(port) do pkt
            unsafe_load(reinterpret(Core.LLVMPtr{UInt64,AS.Global}, pkt))
        end
        hostcall_close!(port)
        out[1] = v
        return
    end
    tt = Tuple{HostcallClient, CuDeviceVector{UInt64,1}}
    modern = sprint(io -> CUDA.code_ptx(io, probe, tt; arch=sm"70"))
    @test occursin("ld.relaxed.sys.global.u32", modern)
    @test occursin("st.relaxed.sys.global.u32", modern)
    @test occursin("fence.acq_rel.sys", modern)
    @test occursin("nanosleep", modern)

    legacy = sprint(io -> CUDA.code_ptx(io, probe, tt; arch=sm"60"))
    @test occursin("ld.volatile.global.u32", legacy)
    @test occursin("st.volatile.global.u32", legacy)
    @test occursin("membar.sys", legacy)
    @test !occursin("nanosleep", legacy)

    for ptx in (modern, legacy)
        @test occursin(r"atom\.(global\.)?or\.b32", ptx)
        @test occursin("activemask", ptx)
        @test occursin("bar.warp.sync", ptx)
    end
end
