# support for device-side exceptions

## exception type

struct KernelException <: Exception
    dev::CuDevice
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device $(name(err.dev))")
end


## exception handling

# the exception flag of every context that kernels were compiled for. these are checked on
# every synchronization, without locking, so the list is never modified, only replaced.
mutable struct ExceptionFlags
    Base.@atomic entries::Vector{Tuple{CuContext,HostMemory}}
end
const _exception_flags = ExceptionFlags(Tuple{CuContext,HostMemory}[])
const exception_flags_lock = ReentrantLock()

exception_flags() = Base.@atomic _exception_flags.entries

# get the exception flag of the module's context, which kernels use to signal errors to the
# host, creating it on first use. an existing flag is left alone: it may already hold an
# exception of a kernel that ran before this module was loaded, which the next
# synchronization should still report.
function create_exceptions!(mod::CuModule)
    mem = @lock exception_flags_lock begin
        entries = exception_flags()
        i = findfirst(((ctx, _),) -> ctx == mod.ctx, entries)
        if i === nothing
            mem = alloc(HostMemory, sizeof(ExceptionInfo_st), MEMHOSTALLOC_DEVICEMAP)
            unsafe_store!(convert(ExceptionInfo, mem), ExceptionInfo_st())
            Base.@atomic _exception_flags.entries = push!(copy(entries), (mod.ctx, mem))
            mem
        else
            entries[i][2]
        end
    end
    return convert(ExceptionInfo, mem)
end

# destroying a context frees its exception flag
function forget_exceptions!(ctx::CuContext)
    @lock exception_flags_lock begin
        entries = filter(((other, _),) -> other != ctx, exception_flags())
        Base.@atomic _exception_flags.entries = entries
    end
    return
end

# check the exception flags on every API call, similarly to how CUDA handles errors
function check_exceptions()
    for (ctx,mem) in exception_flags()
        # (how the driver reports a destroyed context's handle can change over time, so
        #  stop checking it)
        if !isvalid(ctx)
            forget_exceptions!(ctx)
            continue
        end
        exception_info = convert(ExceptionInfo, mem)
        if exception_info.status != 0
            # restore the structure
            unsafe_store!(exception_info, ExceptionInfo_st())

            # flush the kernel's diagnostic output before reporting the exception
            res = flush_output(ctx)
            res == SUCCESS || throw_api_error(res)

            # throw host-side
            dev = device(ctx)
            throw(KernelException(dev))
        end
    end
    return
end
