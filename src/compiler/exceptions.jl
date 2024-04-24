# support for device-side exceptions

## exception type

struct KernelException <: Exception
    dev::CuDevice
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device $(name(err.dev))")
end


## exception handling

const exception_flags = Dict{CuContext, HostMemory}()

# create a CPU/GPU exception flag for error signalling, and put it in the module
function create_exceptions!(mod::CuModule)
    mem = get!(exception_flags, mod.ctx) do
        alloc(HostMemory, sizeof(UInt8), MEMHOSTALLOC_DEVICEMAP)
    end
    ptr = convert(Ptr{UInt8}, mem)
    unsafe_store!(ptr, 0)
    return ptr
end

# check the exception flags on every API call, similarly to how CUDA handles errors
function check_exceptions()
    for (ctx,mem) in exception_flags
        if isvalid(ctx)
            ptr = convert(Ptr{UInt8}, mem)
            flag = unsafe_load(ptr)
            if flag != 0
                unsafe_store!(ptr, 0)
                dev = device(ctx)
                throw(KernelException(dev))
            end
        end
    end
    return
end
