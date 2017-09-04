const RUNTIME_FUNCTION = "calls the Julia runtime"
const UNKNOWN_FUNCTION = "calls an unknown function"

struct InvalidIRError <: Exception
    kind::String
    meta::Any
end

InvalidIRError(kind) = InvalidIRError(kind, nothing)

function validate_ir!(errors::Vector{>:InvalidIRError}, mod::LLVM.Module)
    for f in functions(mod)
        validate_ir!(errors, f)
    end
    return errors
end

function validate_ir!(errors::Vector{>:InvalidIRError}, f::LLVM.Function)
    @trace("Validating $(LLVM.name(f))")

    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            validate_ir!(errors, inst)
        end
    end

    return errors
end

const special_fns = ["vprintf", "__nvvm_reflect"]

function validate_ir!(errors::Vector{>:InvalidIRError}, inst::LLVM.CallInst)
    dest_f = called_value(inst)
    dest_fn = LLVM.name(dest_f)

    runtime = Libdl.dlopen("libjulia")
    if isa(dest_f, GlobalValue)
        if isdeclaration(dest_f) && intrinsic_id(dest_f) == 0 && !(dest_fn in special_fns)
            if Libdl.dlsym_e(runtime, dest_fn) != C_NULL
                push!(errors, InvalidIRError(RUNTIME_FUNCTION, (dest_fn, inst)))
            else
                push!(errors, InvalidIRError(UNKNOWN_FUNCTION, (dest_f, inst)))
            end
        end
    elseif isa(dest_f, InlineAsm)
        # let's assume it's valid ASM
    end

    return errors
end

validate_ir(args...) = validate_ir!(Vector{InvalidIRError}(), args...)
