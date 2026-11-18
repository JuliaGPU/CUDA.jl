# support for device-side exceptions

## exception type

"""
    KernelException

An exception thrown during kernel execution on device `dev`, by the thread with index
`thread` in the block with index `block`. If the kernel was compiled at debug level 1 or
higher (see the `debug_level` option of `@cuda`, and Julia's `-g` option), `site`
describes what was thrown: the exception's name, the reason, and at debug level 2 a
backtrace.
"""
struct KernelException <: Exception
    dev::CuDevice
    site::Union{Nothing,GPUCompiler.ExceptionSite}
    thread::NTuple{3,Int}
    block::NTuple{3,Int}
end

function Base.showerror(io::IO, err::KernelException)
    site = err.site
    print(io, "KernelException: ", site === nothing ? "exception thrown" : site.name,
          " during kernel execution on thread ", err.thread, " in block ", err.block,
          " of device ", name(err.dev))
    site === nothing && return
    isempty(site.reason) || print(io, ": ", site.reason)
    if isempty(site.frames)
        print(io, "\nStacktrace not available, launch the kernel with `@cuda debug_level=2` or run Julia with -g2 for more details.")
    else
        print(io, "\nStacktrace:")
        for (i, frame) in enumerate(site.frames)
            print(io, "\n [", i, "] ", frame.func, " at ", frame.file, ":", frame.line)
        end
    end
end


## exception handling

# the exception flag of every context that kernels were compiled for, with the tables of
# throw sites of those kernels, which the flag refers to. the flags are checked on every
# synchronization, without locking, so the list is never modified, only replaced. the tables
# are only added to, and only accessed, with the lock held.
const ExceptionFlag = Tuple{CuContext,HostMemory,Vector{Vector{GPUCompiler.ExceptionSite}}}
mutable struct ExceptionFlags
    Base.@atomic entries::Vector{ExceptionFlag}
end
const _exception_flags = ExceptionFlags(ExceptionFlag[])
const exception_flags_lock = ReentrantLock()

exception_flags() = Base.@atomic _exception_flags.entries

# get the exception flag of the module's context, which kernels use to signal errors to the
# host, creating it on first use, and register the table of throw sites of a kernel in the
# module. returns the flag, and the index of the table that the kernel reports exceptions
# with (0 if it has no throw sites). an existing flag is left alone: it may already hold an
# exception of a kernel that ran before this module was loaded, which the next
# synchronization should still report.
function create_exceptions!(mod::CuModule, sites::Vector{GPUCompiler.ExceptionSite})
    mem, index = @lock exception_flags_lock begin
        entries = exception_flags()
        i = findfirst(((ctx, _),) -> ctx == mod.ctx, entries)
        if i === nothing
            mem = alloc(HostMemory, sizeof(ExceptionInfo_st), MEMHOSTALLOC_DEVICEMAP)
            unsafe_store!(convert(ExceptionInfo, mem), ExceptionInfo_st())
            tables = Vector{GPUCompiler.ExceptionSite}[]
            Base.@atomic _exception_flags.entries = push!(copy(entries), (mod.ctx, mem, tables))
        else
            _, mem, tables = entries[i]
        end
        isempty(sites) || push!(tables, sites)
        mem, isempty(sites) ? UInt32(0) : length(tables) % UInt32
    end
    return convert(ExceptionInfo, mem), index
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
    for (ctx, mem, tables) in exception_flags()
        # (how the driver reports a destroyed context's handle can change over time, so
        #  stop checking it)
        if !isvalid(ctx)
            forget_exceptions!(ctx)
            continue
        end
        exception_info = convert(ExceptionInfo, mem)
        if exception_info.status != 0
            throw_kernel_exception(ctx, exception_info, tables)
        end
    end
    return
end

@noinline function throw_kernel_exception(ctx::CuContext, exception_info::ExceptionInfo,
                                          tables::Vector{Vector{GPUCompiler.ExceptionSite}})
    # the kernel sets the status after recording the exception
    info = unsafe_load(exception_info)

    # restore the structure
    unsafe_store!(exception_info, ExceptionInfo_st())

    # flush the kernel's output before reporting the exception
    res = flush_output(ctx)
    res == SUCCESS || throw_api_error(res)

    site = @lock exception_flags_lock begin
        table = get(tables, info.sites, nothing)
        table === nothing ? nothing : get(table, info.site, nothing)
    end
    thread = Int.(values(info.threadIdx))
    block = Int.(values(info.blockIdx))
    throw(KernelException(device(ctx), site, thread, block))
end
