# support for device-side exceptions

## exception type

struct KernelException <: Exception
    dev::CuDevice
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device $(name(err.dev))")
end


## exception codegen

# emit a global variable for storing the current exception status
#
# since we don't actually support globals, access to this variable is done by calling the
# julia_exception_flag function (lowered here to actual accesses of the variable)
function emit_exception_flag!(mod::LLVM.Module)
    @assert !haskey(globals(mod), "exception_flag")

    # add the global variable
    T_ptr = convert(LLVMType, Ptr{Cvoid})
    gv = GlobalVariable(mod, T_ptr, "exception_flag")
    initializer!(gv, LLVM.ConstantInt(T_ptr, 0))
    linkage!(gv, LLVM.API.LLVMWeakAnyLinkage)
    extinit!(gv, true)
    set_used!(mod, gv)

    # lower uses of the getter
    if haskey(functions(mod), "julia_exception_flag")
        buf_getter = functions(mod)["julia_exception_flag"]
        @assert return_type(eltype(llvmtype(buf_getter))) == eltype(llvmtype(gv))

        # find uses
        worklist = Vector{LLVM.CallInst}()
        for use in uses(buf_getter)
            call = user(use)::LLVM.CallInst
            push!(worklist, call)
        end

        # replace uses by a load from the global variable
        for call in worklist
            Builder(JuliaContext()) do builder
                position!(builder, call)
                ptr = load!(builder, gv)
                replace_uses!(call, ptr)
            end
            unsafe_delete!(LLVM.parent(call), call)
        end
    end
end


## exception handling

const exception_flags = Dict{CuContext, Mem.HostBuffer}()

# create a CPU/GPU exception flag for error signalling, and put it in the module
#
# also see compiler/irgen.jl::emit_exception_flag!
function create_exceptions!(mod::CuModule)
    if VERSION >= v"1.5" && !(v"1.6-" <= VERSION < v"1.6.0-DEV.90")
        flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, "exception_flag")
        exception_flag = get!(exception_flags, mod.ctx,
                            Mem.alloc(Mem.Host, sizeof(Int), Mem.HOSTALLOC_DEVICEMAP))
        flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, exception_flag))
    else
        ctx = mod.ctx
        try
            flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, "exception_flag")
            exception_flag = get!(exception_flags, ctx, Mem.alloc(Mem.Host, sizeof(Int),
                                Mem.HOSTALLOC_DEVICEMAP))
            flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, exception_flag))
        catch err
            (isa(err, CuError) && err.code == ERROR_NOT_FOUND) || rethrow()
        end
    end

    return
end

# check the exception flags on every API call, similarly to how CUDA handles errors
function check_exceptions()
    for (ctx,buf) in exception_flags
        if isvalid(ctx)
            ptr = convert(Ptr{Int}, buf)
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
