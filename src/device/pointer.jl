# CUDA-specific operations on pointers with address spaces

## adrspace aliases

export AS

module AS

const Generic  = 0
const Global   = 1
const Shared   = 3
const Constant = 4
const Local    = 5

end


## ldg

# operand types supported by llvm.nvvm.ldg.global
# NOTE: CUDA 8.0 supports more caching modifiers, but those aren't supported by LLVM yet
const LDGTypes = Union{UInt8, UInt16, UInt32, UInt64,
                       Int8, Int16, Int32, Int64,
                       Float32, Float64}

# TODO: this functionality should throw <sm_32
@generated function pointerref_ldg(p::LLVMPtr{T,AS.Global}, i::Int,
                                   ::Val{align}) where {T<:LDGTypes,align}
    sizeof(T) == 0 && return T.instance
    JuliaContext() do ctx
        eltyp = convert(LLVMType, T, ctx)

        # TODO: ccall the intrinsic directly

        T_int = convert(LLVMType, Int, ctx)
        T_int32 = LLVM.Int32Type(ctx)
        T_ptr = convert(LLVMType, p, ctx)
        T_typed_ptr = LLVM.PointerType(eltyp, AS.Global)

        # create a function
        param_types = [T_ptr, T_int]
        llvm_f, _ = create_function(eltyp, param_types)

        # create the intrinsic
        intrinsic_name = let
            class = if isa(eltyp, LLVM.IntegerType)
                :i
            elseif isa(eltyp, LLVM.FloatingPointType)
                :f
            else
                error("Cannot handle $eltyp argument to unsafe_cached_load")
            end
            width = sizeof(T)*8
            typ = Symbol(class, width)
            "llvm.nvvm.ldg.global.$class.$typ.p1$typ"
        end
        mod = LLVM.parent(llvm_f)
        intrinsic_typ = LLVM.FunctionType(eltyp, [T_typed_ptr, T_int32])
        intrinsic = LLVM.Function(mod, intrinsic_name, intrinsic_typ)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = bitcast!(builder, parameters(llvm_f)[1], T_typed_ptr)
            typed_ptr = gep!(builder, typed_ptr, [parameters(llvm_f)[2]])
            ld = call!(builder, intrinsic,
                    [typed_ptr, ConstantInt(Int32(align), ctx)])

            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global, ctx)

            ret!(builder, ld)
        end

        call_function(llvm_f, T, Tuple{LLVMPtr{T,AS.Global}, Int}, :((p, Int(i-one(i)))))
    end
end

# interface

export unsafe_cached_load

unsafe_cached_load(p::LLVMPtr{<:LDGTypes,1}, i::Integer=1, align::Val=Val(1)) =
    pointerref_ldg(p, Int(i), align)
# NOTE: fall back to normal unsafe_load for unsupported types. we could be smarter here,
#       e.g. destruct/load/reconstruct, but that's too complicated for what it's worth.
unsafe_cached_load(p::LLVMPtr, i::Integer=1, align::Val=Val(1)) =
    unsafe_load(p, i, align)
