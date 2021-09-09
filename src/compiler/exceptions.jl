# support for device-side exceptions

## exception type

struct KernelException <: Exception
    dev::CuDevice
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device $(name(err.dev))")
end


## exception handling

const exception_flags = Dict{CuContext, Mem.HostBuffer}()

# create a CPU/GPU exception flag for error signalling, and put it in the module
function create_exceptions!(mod::CuModule)
    exception_flag = get!(exception_flags, mod.ctx,
                          Mem.alloc(Mem.Host, sizeof(Int8), Mem.HOSTALLOC_DEVICEMAP))
    return reinterpret(LLVMPtr{Int8, AS.Global}, convert(CuPtr{Int8}, exception_flag))
end

# check the exception flags on every API call, similarly to how CUDA handles errors
function check_exceptions()
    for (ctx,buf) in exception_flags
        if isvalid(ctx)
            ptr = convert(Ptr{Int8}, buf)
            flag = unsafe_load(ptr)
            if flag != 0
                unsafe_store!(ptr, 0)
                dev = CuDevice(ctx)
                throw(KernelException(dev))
            end
        end
    end
    return
end
