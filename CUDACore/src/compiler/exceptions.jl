# support for device-side exceptions

## exception type

# what the device reported about an exception (see `ExceptionReport` in the runtime)
struct KernelExceptionInfo
    name::String
    subtype::String
    reason::String
    thread::NTuple{3,Int}
    block::NTuple{3,Int}
    stacktrace::Union{Nothing,Vector{@NamedTuple{idx::Int, func::String, file::String, line::Int}}}
end

struct KernelException <: Exception
    dev::CuDevice
    info::Union{Nothing,KernelExceptionInfo}
end
KernelException(dev::CuDevice) = KernelException(dev, nothing)

function Base.showerror(io::IO, err::KernelException)
    info = err.info
    if info === nothing
        print(io, "KernelException: exception thrown during kernel execution on device $(name(err.dev))")
        return
    end
    kind = isempty(info.subtype) ? info.name : info.subtype
    isempty(kind) && (kind = "exception")
    article = lowercase(first(kind)) in "aeiou" ? "an" : "a"
    print(io, "KernelException: ", article, " ", kind,
          " was thrown during kernel execution on device ", name(err.dev),
          ", thread ", info.thread, " in block ", info.block, ".")
    isempty(info.reason) || print(io, "\n", info.reason)
    if info.stacktrace === nothing
        print(io, "\nStacktrace not available, run Julia on debug level 2 for more details (by passing -g2 to the executable).")
    else
        print(io, "\nStacktrace:")
        for frame in sort(info.stacktrace; by=f->f.idx)
            print(io, "\n [", frame.idx, "] ", frame.func, " at ", frame.file, ":", frame.line)
        end
    end
end

# reports received through the hostcall area, keyed by context and reporting thread;
# frames may arrive in any order and are assembled on the host
const kernel_exception_reports = Dict{CuContext, Dict{NTuple{6,Int}, KernelExceptionInfo}}()
const kernel_exception_reports_lock = Threads.SpinLock()

# copy a NUL-terminated string from device memory on the given stream. the string is a
# module global, so we cannot read past the allocation that contains it.
function device_string(ptr::Ptr{UInt8}, stream::CuStream)
    ptr == C_NULL && return ""
    dptr = reinterpret(CuPtr{UInt8}, ptr)
    base = Ref{CuPtr{Cvoid}}()
    size = Ref{Csize_t}()
    res = unchecked_cuMemGetAddressRange_v2(base, size, dptr)
    res == SUCCESS || return "<invalid string pointer>"
    avail = Int(size[]) - (Int(dptr) - Int(base[]))
    nbytes = clamp(avail, 0, 4096)
    nbytes == 0 && return ""
    buf = Vector{UInt8}(undef, nbytes)
    res = unchecked_cuMemcpyDtoHAsync_v2(buf, dptr, nbytes, stream)
    res == SUCCESS || return "<invalid string pointer>"
    cuStreamSynchronize(stream)
    n = findfirst(iszero, buf)
    return String(n === nothing ? buf : buf[1:n-1])
end

# the built-in handler for exception reports (registered in hostcall.jl); runs on the
# server thread (or in a draining task) with the context active and the hostcall stream
function service_exception_report(p, hdr)
    area = p.area
    mask = hdr.mask
    for lane in 0:31
        (mask >> lane) & 1 == 0 && continue
        report = unsafe_load(convert(Ptr{ExceptionReport}, lane_packet(p, lane)))
        key = (Int(report.block.x), Int(report.block.y), Int(report.block.z),
               Int(report.thread.x), Int(report.thread.y), Int(report.thread.z))
        stream = hostcall_stream(area)
        if report.kind == EXCEPTION_REPORT_FRAME
            frame = (; idx=Int(report.idx), func=device_string(report.a, stream),
                       file=device_string(report.b, stream), line=Int(report.line))
            @lock kernel_exception_reports_lock begin
                reports = get!(Dict{NTuple{6,Int}, KernelExceptionInfo}, kernel_exception_reports, area.ctx)
                info = get(reports, key, nothing)
                if info === nothing
                    # the frame arrived before the report; create a placeholder
                    info = KernelExceptionInfo("", "", "", key[4:6], key[1:3], [frame])
                    reports[key] = info
                elseif info.stacktrace !== nothing
                    push!(info.stacktrace, frame)
                end
            end
        else
            name = device_string(report.a, stream)
            subtype = device_string(report.b, stream)
            reason = device_string(report.c, stream)
            @lock kernel_exception_reports_lock begin
                reports = get!(Dict{NTuple{6,Int}, KernelExceptionInfo}, kernel_exception_reports, area.ctx)
                frames = if report.kind == EXCEPTION_REPORT_NAME
                    prev = get(reports, key, nothing)
                    prev === nothing || prev.stacktrace === nothing ? @NamedTuple{idx::Int, func::String, file::String, line::Int}[] : prev.stacktrace
                else
                    nothing
                end
                reports[key] = KernelExceptionInfo(name, subtype, reason, key[4:6], key[1:3], frames)
            end
        end
    end
    hport_flip!(p)
    return
end

# take the reports of a context, if any
function take_exception_reports!(ctx::CuContext)
    @lock kernel_exception_reports_lock begin
        reports = get(kernel_exception_reports, ctx, nothing)
        reports === nothing && return nothing
        delete!(kernel_exception_reports, ctx)
        return reports
    end
end


## exception handling

const exception_infos = Dict{CuContext, HostMemory}()
const exception_infos_lock = ReentrantLock()

# A no-hostcall descriptor follows the exception flag in the same mapped allocation. This
# gives every kernel one runtime-state pointer, even when hostcalls are unavailable.
const exception_client_offset = cld(sizeof(ExceptionInfo_st), sizeof(UInt)) * sizeof(UInt)
const exception_state_size = exception_client_offset + sizeof(HostcallClient)

# create a CPU/GPU exception flag for error signalling, and put it in the module
function create_exceptions!(mod::CuModule)
    mem = @lock exception_infos_lock begin
        get!(exception_infos, mod.ctx) do
            alloc(HostMemory, exception_state_size, MEMHOSTALLOC_DEVICEMAP)
        end
    end
    exception_info = convert(ExceptionInfo, mem)
    unsafe_store!(exception_info, ExceptionInfo_st())
    base = convert(Ptr{UInt8}, mem)
    client_ptr = base + exception_client_offset
    unsafe_store!(convert(Ptr{HostcallClient}, client_ptr),
                  HostcallClient(reinterpret(Ptr{Cvoid}, exception_info)))
    client = reinterpret(HostcallClientPtr, client_ptr)
    return exception_info, client
end

# check the exception flags of a context after synchronizing it, similarly to how CUDA
# handles errors. exceptions are reported per context: synchronizing one device does not
# surface exceptions from kernels on another device, and the flag memory of a context that
# has been destroyed (`unsafe_reset!`) is never touched again.
function check_exceptions(ctx::CuContext=context())
    hostcall_synchronize(ctx)
    mem = @lock exception_infos_lock get(exception_infos, ctx, nothing)
    mem === nothing && return
    exception_info = convert(ExceptionInfo, mem)
    if exception_info.status != 0
        # restore the structure
        unsafe_store!(exception_info, ExceptionInfo_st())

        # pick up the report the device sent before setting the flag
        hostcall_drain()
        reports = take_exception_reports!(ctx)

        # throw host-side
        dev = device(ctx)
        info = reports === nothing || isempty(reports) ? nothing : first(values(reports))
        throw(KernelException(dev, info))
    end
    return
end

# Drop exception state before destroying a context. The flag is raw pinned memory and must
# be released explicitly; reports for a destroyed context can no longer be observed.
function forget_exceptions!(pred)
    forgotten = @lock exception_infos_lock begin
        entries = Pair{CuContext,HostMemory}[]
        for (ctx, mem) in collect(exception_infos)
            pred(ctx) || continue
            push!(entries, ctx => mem)
            delete!(exception_infos, ctx)
        end
        entries
    end
    for (_, mem) in forgotten
        free(mem)
    end
    @lock kernel_exception_reports_lock begin
        for ctx in collect(keys(kernel_exception_reports))
            pred(ctx) && delete!(kernel_exception_reports, ctx)
        end
    end
    return
end
forget_exceptions!(ctx::CuContext) = forget_exceptions!(candidate -> candidate == ctx)
forget_exceptions!(dev::CuDevice) = forget_exceptions!(ctx -> device(ctx) == dev)
