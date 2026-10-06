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

# create a CPU/GPU exception flag for error signalling, and put it in the module
function create_exceptions!(mod::CuModule)
    mem = get!(exception_infos, mod.ctx) do
        alloc(HostMemory, sizeof(ExceptionInfo_st), MEMHOSTALLOC_DEVICEMAP)
    end
    exception_info = convert(ExceptionInfo, mem)
    unsafe_store!(exception_info, ExceptionInfo_st())
    return exception_info
end

# check the exception flags on every API call, similarly to how CUDA handles errors
function check_exceptions()
    for (ctx,mem) in exception_infos
        exception_info = convert(ExceptionInfo, mem)
        if exception_info.status != 0
            # restore the structure
            unsafe_store!(exception_info, ExceptionInfo_st())

            # flush the output of the kernel, which querying for completion doesn't do (as
            # non-blocking synchronization does). synchronizing an event that was never
            # recorded doesn't wait for anything, also not for unrelated work. that's also
            # allowed while capturing, once relaxing the capture mode.
            context!(ctx) do
                event = CuEvent(EVENT_DISABLE_TIMING)
                relaxed_capture_mode(() -> cuEventSynchronize(event))
            end

            # throw host-side
            dev = device(ctx)
            throw(KernelException(dev))
        end
    end
    return
end
