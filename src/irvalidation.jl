const UNKNOWN_FUNCTION = "unknown function"

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
            dest_f = called_value(inst)
            if isa(dest_f, GlobalValue)
                if isdeclaration(dest_f) && intrinsic_id(dest_f) == 0
                    push!(errors, InvalidIRError(UNKNOWN_FUNCTION, (inst, dest_f)))
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
