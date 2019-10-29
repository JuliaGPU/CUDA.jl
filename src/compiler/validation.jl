# validation of properties and code

function check_method(job::CompilerJob)
    isa(job.f, Core.Builtin) && throw(KernelError(job, "function is not a generic function"))

    # get the method
    ms = Base.methods(job.f, job.tt)
    isempty(ms)   && throw(KernelError(job, "no method found"))
    length(ms)!=1 && throw(KernelError(job, "no unique matching method"))
    m = first(ms)

    # kernels can't return values
    if job.kernel
        rt = Base.return_types(job.f, job.tt)[1]
        if rt != Nothing
            throw(KernelError(job, "kernel returns a value of type `$rt`",
                """Make sure your kernel function ends in `return`, `return nothing` or `nothing`.
                   If the returned value is of type `Union{}`, your Julia code probably throws an exception.
                   Inspect the code with `@device_code_warntype` for more details."""))
        end
    end

    return
end

if VERSION < v"1.1.0-DEV.593"
    fieldtypes(@nospecialize(dt)) = ntuple(i->fieldtype(dt, i), fieldcount(dt))
end

# The actual check is rather complicated
# and might change from version to version...
function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

function explain_nonisbits(@nospecialize(dt), depth=1)
    hasfieldcount(dt) || return ""
    msg = ""
    for (ft, fn) in zip(fieldtypes(dt), fieldnames(dt))
        if !isbitstype(ft)
            msg *= "  "^depth * ".$fn is of type $ft which is not isbits.\n"
            msg *= explain_nonisbits(ft, depth+1)
        end
    end
    return msg
end

function check_invocation(job::CompilerJob, entry::LLVM.Function)
    # make sure any non-isbits arguments are unused
    real_arg_i = 0
    sig = Base.signature_type(job.f, job.tt)::Type
    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        real_arg_i += 1

        if !isbitstype(dt)
            param = parameters(entry)[real_arg_i]
            if !isempty(uses(param))
                msg = """Argument $arg_i to your kernel function is of type $dt.
                       That type is not isbits, and such arguments are only allowed when they are unused by the kernel."""

                # explain which fields are not isbits
                msg *= explain_nonisbits(dt)

                throw(KernelError(job, "passing and using non-bitstype argument", msg))
            end
        end
    end

    return
end


## IR validation

const IRError = Tuple{String, StackTraces.StackTrace, Any} # kind, bt, meta

struct InvalidIRError <: Exception
    job::CompilerJob
    errors::Vector{IRError}
end

const RUNTIME_FUNCTION = "call to the Julia runtime"
const UNKNOWN_FUNCTION = "call to an unknown function"
const POINTER_FUNCTION = "call through a literal pointer"
const DELAYED_BINDING  = "use of an undefined name"
const DYNAMIC_CALL     = "dynamic function invocation"

function Base.showerror(io::IO, err::InvalidIRError)
    print(io, "InvalidIRError: compiling $(signature(err.job)) resulted in invalid LLVM IR")
    for (kind, bt, meta) in err.errors
        print(io, "\nReason: unsupported $kind")
        if meta != nothing
            if kind == RUNTIME_FUNCTION || kind == UNKNOWN_FUNCTION || kind == POINTER_FUNCTION || kind == DYNAMIC_CALL
                print(io, " (call to ", meta, ")")
            elseif kind == DELAYED_BINDING
                print(io, " (use of '", meta, "')")
            end
        end
        Base.show_backtrace(io, bt)
    end
    return
end

function check_ir(job, args...)
    errors = check_ir!(job, IRError[], args...)
    unique!(errors)
    if !isempty(errors)
        throw(InvalidIRError(job, errors))
    end

    return
end

function check_ir!(job, errors::Vector{IRError}, mod::LLVM.Module)
    for f in functions(mod)
        check_ir!(job, errors, f)
    end

    return errors
end

function check_ir!(job, errors::Vector{IRError}, f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            check_ir!(job, errors, inst)
        end
    end

    return errors
end

const special_fns = (
    # PTX intrinsics
    "vprintf", "__assertfail", "malloc", "free",
    # libdevice
    "__nvvm_reflect",
    # libcudevrt
    "cudaDeviceSynchronize", "cudaGetParameterBufferV2", "cudaLaunchDeviceV2",
    "cudaCGGetIntrinsicHandle", "cudaCGSynchronize"
)

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.CallInst)
    dest = called_value(inst)
    if isa(dest, LLVM.Function)
        fn = LLVM.name(dest)

        # some special handling for runtime functions that we don't implement

        if fn == "jl_get_binding_or_error"
            # interpret the arguments
            sym = try
                m, sym, _ = operands(inst)
                sym = first(operands(sym::ConstantExpr))::ConstantInt
                sym = convert(Int, sym)
                sym = Ptr{Cvoid}(sym)
                Base.unsafe_pointer_to_objref(sym)
            catch e
                isa(e,TypeError) || rethrow()
                @warn "Decoding arguments to jl_get_binding_or_error failed, please file a bug with a reproducer." inst bb=LLVM.parent(inst)
                nothing
            end

            if sym !== nothing
                bt = backtrace(inst)
                push!(errors, (DELAYED_BINDING, bt, sym))
                return errors
            end

        elseif fn == "jl_invoke"
            # interpret the arguments
            meth = try
                if VERSION < v"1.3.0-DEV.244"
                    meth, args, nargs, _ = operands(inst)
                else
                    f, args, nargs, meth = operands(inst)
                end
                meth = first(operands(meth::ConstantExpr))::ConstantExpr
                meth = first(operands(meth))::ConstantInt
                meth = convert(Int, meth)
                meth = Ptr{Cvoid}(meth)
                Base.unsafe_pointer_to_objref(meth)::Core.MethodInstance
            catch e
                isa(e,TypeError) || rethrow()
                @warn "Decoding arguments to jl_invoke failed, please file a bug with a reproducer." inst bb=LLVM.parent(inst)
                nothing
            end

            if meth !== nothing
                bt = backtrace(inst)
                push!(errors, (DYNAMIC_CALL, bt, meth.def))
                return errors
            end

        elseif fn == "jl_apply_generic"
            # interpret the arguments
            f = try
                if VERSION < v"1.3.0-DEV.244"
                    args, nargs, _ = operands(inst)
                    ## args is a buffer where arguments are stored in
                    f, args = user.(uses(args))
                    ## first store into the args buffer is a direct store
                    f = first(operands(f::LLVM.StoreInst))::ConstantExpr
                else
                    f, args, nargs, _ = operands(inst)
                end

                f = first(operands(f))::ConstantExpr # get rid of addrspacecast
                f = first(operands(f))::ConstantInt # get rid of inttoptr
                f = convert(Int, f)
                f = Ptr{Cvoid}(f)
                Base.unsafe_pointer_to_objref(f)
            catch e
                isa(e,TypeError) || rethrow()
                @warn "Decoding arguments to jl_apply_generic failed, please file a bug with a reproducer." inst bb=LLVM.parent(inst)
                nothing
            end

            if f !== nothing
                bt = backtrace(inst)
                push!(errors, (DYNAMIC_CALL, bt, f))
                return errors
            end
        end

        # detect calls to undefined functions
        if isdeclaration(dest) && intrinsic_id(dest) == 0 && !(fn in special_fns)
            # figure out if the function lives in the Julia runtime library
            if libjulia[] == C_NULL
                paths = filter(Libdl.dllist()) do path
                    name = splitdir(path)[2]
                    startswith(name, "libjulia")
                end
                libjulia[] = Libdl.dlopen(first(paths))
            end

            bt = backtrace(inst)
            if Libdl.dlsym_e(libjulia[], fn) != C_NULL
                push!(errors, (RUNTIME_FUNCTION, bt, LLVM.name(dest)))
            else
                push!(errors, (UNKNOWN_FUNCTION, bt, LLVM.name(dest)))
            end
        end
    elseif isa(dest, InlineAsm)
        # let's assume it's valid ASM
    elseif isa(dest, ConstantExpr)
        # detect calls to literal pointers
        if occursin("inttoptr", string(dest))
            # extract the literal pointer
            ptr_arg = first(operands(dest))
            @compiler_assert isa(ptr_arg, ConstantInt) job
            ptr_val = convert(Int, ptr_arg)
            ptr = Ptr{Cvoid}(ptr_val)

            # look it up in the Julia JIT cache
            bt = backtrace(inst)
            frames = ccall(:jl_lookup_code_address, Any, (Ptr{Cvoid}, Cint,), ptr, 0)
            if length(frames) >= 1
                @compiler_assert length(frames) == 1 job frames=frames
                if VERSION >= v"1.4.0-DEV.123"
                    fn, file, line, linfo, fromC, inlined = last(frames)
                else
                    fn, file, line, linfo, fromC, inlined, ip = last(frames)
                end
                push!(errors, (POINTER_FUNCTION, bt, fn))
            else
                push!(errors, (POINTER_FUNCTION, bt, nothing))
            end
        end
    end

    return errors
end
