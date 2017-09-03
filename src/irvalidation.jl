const RUNTIME_FUNCTION = "contains call to the runtime"
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
    runtime = Libdl.dlopen("libjulia")

    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            dest_f = called_value(inst)
            dest_fn = LLVM.name(dest_f)
            if isa(dest_f, GlobalValue)
                if isdeclaration(dest_f) && intrinsic_id(dest_f) == 0
                    if Libdl.dlsym_e(runtime, dest_fn) != C_NULL
                    push!(errors, InvalidIRError(RUNTIME_FUNCTION, (inst, dest_f)))
                else
                    push!(errors, InvalidIRError(UNKNOWN_FUNCTION, (inst, dest_f)))
                end
                end
            elseif isa(dest_f, InlineAsm)
                # let's assume it's valid ASM
            else
                warn(dest_f)
            end
        end
    end
    return errors
end

validate_ir(args...) = validate_ir!(Vector{InvalidIRError}(), args...)
