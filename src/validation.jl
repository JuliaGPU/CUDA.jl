## validation of properties and code

# check validity of a function invocation
function validate_invocation(@nospecialize(ctx::CompilerContext))
    # get the method
    ms = Base.methods(ctx.f, ctx.tt)
    isempty(ms)   && compiler_error(ctx, "no method found")
    length(ms)!=1 && compiler_error(ctx, "no unique matching method")
    m = first(ms)

    # kernels can't return values
    if ctx.kernel
        rt = Base.return_types(ctx.f, ctx.tt)[1]
        if rt != Nothing
            compiler_error(ctx, "kernel returning a value"; return_type=rt)
        end
    end
end


## IR validation

const RUNTIME_FUNCTION = "call to the Julia runtime"
const UNKNOWN_FUNCTION = "call to an unknown function"

struct UnsupportedIRError <: Exception
    kind::String
    compiletrace::StackTraces.StackTrace
    meta::Any
end

UnsupportedIRError(kind) = UnsupportedIRError(kind, nothing)

function Base.showerror(io::IO, err::UnsupportedIRError)
    print(io, "unsupported $(err.kind)")
    if err.kind == RUNTIME_FUNCTION || err.kind == UNKNOWN_FUNCTION
        print(io, " (", err.meta, ")")
    end
end

function backtrace(inst)
    name = Ref{Cstring}()
    filename = Ref{Cstring}()
    line = Ref{Cuint}()
    col = Ref{Cuint}()

    depth = 0
    stack = StackTraces.StackFrame[]
    while LLVM.API.LLVMGetSourceLocation(LLVM.ref(inst), depth, name, filename, line, col) == 1
        frame = StackTraces.StackFrame(replace(unsafe_string(name[]), r";$"=>""), unsafe_string(filename[]), line[])
        push!(stack, frame)
        depth += 1
    end

    if last(stack).func == :KernelWrapper
        pop!(stack)
    end

    stack
end

function validate_ir!(errors::Vector{>:UnsupportedIRError}, mod::LLVM.Module)
    for f in functions(mod)
        validate_ir!(errors, f)
    end
    return errors
end

function validate_ir!(errors::Vector{>:UnsupportedIRError}, f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            validate_ir!(errors, inst)
        end
    end

    return errors
end

const special_fns = ["vprintf", "__nvvm_reflect"]

function validate_ir!(errors::Vector{>:UnsupportedIRError}, inst::LLVM.CallInst)
    dest_f = called_value(inst)
    dest_fn = LLVM.name(dest_f)
    lib = first(filter(lib->startswith(lib, "libjulia"), map(path->splitdir(path)[2], Libdl.dllist())))
    runtime = Libdl.dlopen(lib)
    if isa(dest_f, GlobalValue)
        if isdeclaration(dest_f) && intrinsic_id(dest_f) == 0 && !(dest_fn in special_fns)
            compiletrace = backtrace(inst)
            if Libdl.dlsym_e(runtime, dest_fn) != C_NULL
                push!(errors, UnsupportedIRError(RUNTIME_FUNCTION, compiletrace, dest_fn))
            else
                push!(errors, UnsupportedIRError(UNKNOWN_FUNCTION, compiletrace, dest_f))
            end
        end
    elseif isa(dest_f, InlineAsm)
        # let's assume it's valid ASM
    end

    return errors
end

validate_ir(args...) = validate_ir!(Vector{UnsupportedIRError}(), args...)
