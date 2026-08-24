# support for device-side exceptions

## exception type

struct KernelException <: Exception
    dev::CuDevice
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device $(name(err.dev))")
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

        # throw host-side
        dev = device(ctx)
        throw(KernelException(dev))
    end
    return
end

# Drop exception state before destroying a context. The flag is raw pinned memory and must
# be released explicitly.
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
    return
end
forget_exceptions!(ctx::CuContext) = forget_exceptions!(candidate -> candidate == ctx)
forget_exceptions!(dev::CuDevice) = forget_exceptions!(ctx -> device(ctx) == dev)
