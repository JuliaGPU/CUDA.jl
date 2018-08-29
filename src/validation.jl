## validation of properties and code

function check_method(ctx::CompilerContext)
    # get the method
    ms = Base.methods(ctx.f, ctx.tt)
    isempty(ms)   && throw(CompilerError(ctx, "no method found"))
    length(ms)!=1 && throw(CompilerError(ctx, "no unique matching method"))
    m = first(ms)

    # kernels can't return values
    if ctx.kernel
        rt = Base.return_types(ctx.f, ctx.tt)[1]
        if rt != Nothing
            throw(CompilerError(ctx, "kernel returns a value of type $rt"))
        end
    end
end

function check_invocation(ctx::CompilerContext, entry::LLVM.Function)
    # make sure any non-isbits arguments are unused
    real_arg_i = 0
    sig = Base.signature_type(ctx.f, ctx.tt)::Type
    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        real_arg_i += 1

        if !isbitstype(dt)
            param = parameters(entry)[real_arg_i]
            if !isempty(uses(param))
                throw(CompilerError(ctx, "passing and using non-bitstype argument";
                                    argument=arg_i, argument_type=dt))
            end
        end
    end
end


## IR validation

const IRError = Tuple{String, StackTraces.StackTrace, Any} # kind, bt, meta

struct InvalidIRError <: AbstractCompilerError
    ctx::CompilerContext
    errors::Vector{IRError}
end

const RUNTIME_FUNCTION = "call to the Julia runtime"
const UNKNOWN_FUNCTION = "call to an unknown function"

function Base.showerror(io::IO, err::InvalidIRError)
    print(io, "InvalidIRError: compiling $(signature(err.ctx)) resulted in invalid LLVM IR")
    for (kind, bt, meta) in err.errors
        print(io, "\nReason: unsupported $kind")
        if kind == RUNTIME_FUNCTION || kind == UNKNOWN_FUNCTION
            print(io, " (", LLVM.name(meta), ")")
        end
        Base.show_backtrace(io, bt)
    end
end

# generate a pseudo-backtrace from LLVM IR instruction debug information
function backtrace(inst, bt = StackTraces.StackFrame[])
    name = Ref{Cstring}()
    filename = Ref{Cstring}()
    line = Ref{Cuint}()
    col = Ref{Cuint}()

    # look up the debug information from the current instruction
    depth = 0
    while LLVM.API.LLVMGetSourceLocation(LLVM.ref(inst), depth, name, filename, line, col) == 1
        frame = StackTraces.StackFrame(replace(unsafe_string(name[]), r";$"=>""), unsafe_string(filename[]), line[])
        push!(bt, frame)
        depth += 1
    end

    # move up the call chain
    f = LLVM.parent(LLVM.parent(inst))
    callers = uses(f)
    if isempty(callers)
        # wrapping the kernel _does_ trigger a new debug info frame,
        # so just get rid of the wrapper frame.
        if !isempty(bt) && last(bt).func == :KernelWrapper
            pop!(bt)
        end
    else
        # figure out the call sites of this instruction
        call_sites = unique(callers) do call
            # there could be multiple calls, originating from the same source location
            md = metadata(user(call))
            if haskey(md, LLVM.MD_dbg)
                md[LLVM.MD_dbg]
            else
                nothing
            end
        end

        if length(call_sites) > 1
            frame = StackTraces.StackFrame("multiple call sites", "unknown", 0)
            push!(bt, frame)
        elseif length(call_sites) == 1
            backtrace(user(first(call_sites)), bt)
        end
    end

    bt
end

function check_ir!(errors::Vector{IRError}, mod::LLVM.Module)
    for f in functions(mod)
        check_ir!(errors, f)
    end
    return errors
end

function check_ir!(errors::Vector{IRError}, f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            check_ir!(errors, inst)
        end
    end

    return errors
end

const special_fns = ["vprintf", "__nvvm_reflect"]

function check_ir!(errors::Vector{IRError}, inst::LLVM.CallInst)
    dest_f = called_value(inst)
    dest_fn = LLVM.name(dest_f)
    lib = first(filter(lib->startswith(lib, "libjulia"), map(path->splitdir(path)[2], Libdl.dllist())))
    runtime = Libdl.dlopen(lib)
    if isa(dest_f, GlobalValue)
        if isdeclaration(dest_f) && intrinsic_id(dest_f) == 0 && !(dest_fn in special_fns)
            bt = backtrace(inst)
            if Libdl.dlsym_e(runtime, dest_fn) != C_NULL
                push!(errors, (RUNTIME_FUNCTION, bt, dest_f))
            else
                push!(errors, (UNKNOWN_FUNCTION, bt, dest_f))
            end
        end
    elseif isa(dest_f, InlineAsm)
        # let's assume it's valid ASM
    end

    errors
end

function check_ir(ctx, args...)
    errors = check_ir!(IRError[], args...)

    unique!(errors)
    if !isempty(errors)
        throw(InvalidIRError(ctx, errors))
    end
end
