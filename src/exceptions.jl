
# support for device-side exceptions

## exception type

struct KernelException <: Exception
    dev::CuDevice
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device $(CUDAdrv.name(err.dev))")
end

Base.show(io::IO, err::KernelException) = print(io, "KernelException($(err.device))")


## exception handling

const exception_flags = Dict{CuDevice, Mem.HostBuffer}()
push!(device_reset!_listeners, (dev, ctx) -> begin
    # invalidate exception flags when the device resets
    delete!(exception_flags, dev)
end)

# create a CPU/GPU exception flag for error signalling, and put it in the module
#
# also see compiler/irgen.jl::emit_exception_flag!
function create_exceptions!(mod::CuModule)
    ctx = mod.ctx
    dev = device(ctx)
    try
        flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, "exception_flag")
        exception_flag = get!(exception_flags, dev, Mem.alloc(Mem.Host, sizeof(Int),
                            Mem.HOSTALLOC_DEVICEMAP))
        flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, exception_flag))
    catch err
        # modules that do not throw exceptions will not contain the indicator flag
        if err !== CUDAdrv.ERROR_NOT_FOUND
            rethrow()
        end
    end

    CUDAdrv.apicall_hook[] = check_exception_hook

    return
end

function check_exceptions()
    for (dev,buf) in exception_flags
        ptr = convert(Ptr{Int}, buf)
        flag = unsafe_load(ptr)
        if flag !== 0
            unsafe_store!(ptr, 0)
            throw(KernelException(dev))
        end
    end
    return
end

# check the exception flags on every API call, similarly to how CUDA handles errors
function check_exception_hook(apicall)
    # ... but don't do it for some very frequently called functions
    if apicall !== :cuCtxGetCurrent
        check_exceptions()
    end
end
