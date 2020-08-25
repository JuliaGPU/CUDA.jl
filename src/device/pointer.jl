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


## typed ccall

# perform a `ccall(intrinsic, llvmcall)` with accurate pointer types for calling intrinsic,
# which would otherwise cause LLVM assertion failures due to type mismatches.
#
# NOTE: this will become unnecessary when LLVM switches to typeless pointers,
#       or if and when Julia goes back to emitting exact types when passing pointers.
macro typed_ccall(intrinsic, cc, rettyp, argtyps, args...)
    # destructure and validate the arguments
    (intrinsic isa String && startswith(intrinsic, "llvm.")) ||
        error("Can only use @typed_ccall with plain intrinsics")
    cc == :llvmcall || error("Can only use @typed_ccall with the llvmcall calling convention")
    Meta.isexpr(argtyps, :tuple) || error("@typed_ccall expects a tuple of argument types")

    # assign arguments to variables and unsafe_convert/cconvert them as per ccall behavior
    vars = Tuple(gensym() for arg in args)
    var_exprs = map(zip(vars, args)) do (var,arg)
        :($var = $arg)
    end
    arg_exprs = map(zip(vars,argtyps.args)) do (var,typ)
        :(Base.unsafe_convert($typ, Base.cconvert($typ, $var)))
    end

    esc(quote
        $(var_exprs...)
        GC.@preserve $(vars...) begin
            _typed_llvmcall($(Val(Symbol(intrinsic))), $rettyp, Tuple{$(argtyps.args...)}, $(arg_exprs...))
        end
    end)
end

@generated function _typed_llvmcall(::Val{intr}, rettyp, argtt, args...) where {intr}
    # make types available for direct use in this generator
    rettyp = rettyp.parameters[1]
    argtt = argtt.parameters[1]
    argtyps = DataType[argtt.parameters...]
    argexprs = Expr[:(args[$i]) for i in 1:length(args)]

    # build IR that calls the intrinsic, casting types if necessary
    JuliaContext() do ctx
        T_ret = convert(LLVMType, rettyp, ctx)
        T_args = LLVMType[convert(LLVMType, typ, ctx) for typ in argtyps]

        llvm_f, _ = create_function(T_ret, T_args)
        mod = LLVM.parent(llvm_f)

        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            # Julia's compiler strips pointers of their element type.
            # reconstruct those so that we can accurately look up intrinsics.
            T_actual_args = LLVMType[]
            actual_args = LLVM.Value[]
            for (arg, argtyp) in zip(parameters(llvm_f),argtyps)
                if argtyp <: LLVMPtr
                    # passed as i8*
                    T,AS = argtyp.parameters
                    actual_typ = LLVM.PointerType(convert(LLVMType, T, ctx), AS)
                    actual_arg = bitcast!(builder, arg, actual_typ)
                elseif argtyp <: Ptr
                    # passed as i64
                    T = eltype(argtyp)
                    actual_typ = LLVM.PointerType(convert(LLVMType, T, ctx))
                    actual_arg = inttoptr!(builder, arg, actual_typ)
                else
                    actual_typ = convert(LLVMType, argtyp, ctx)
                    actual_arg = arg
                end
                push!(T_actual_args, actual_typ)
                push!(actual_args, actual_arg)
            end

            intr_ft = LLVM.FunctionType(T_ret, T_actual_args)
            intr_f = LLVM.Function(mod, String(intr), intr_ft)

            rv = call!(builder, intr_f, actual_args)

            ret!(builder, rv)
        end

        call_function(llvm_f, rettyp, argtt, :(($(argexprs...),)))
    end
end



## ldg

const LDGTypes = (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64,
                  Float32, Float64)

# TODO: this functionality should throw <sm_32
# NOTE: CUDA 8.0 supports more caching modifiers, but those aren't supported by LLVM yet
for T in LDGTypes
    class = if T <: Integer
        :i
    elseif T <: AbstractFloat
        :f
    end
    # TODO: p class
    width = sizeof(T)*8
    typ = Symbol(class, width)

    intr = "llvm.nvvm.ldg.global.$class.$typ.p1$typ"
    @eval @inline pointerref_ldg(ptr::LLVMPtr{$T,AS.Global}, i::Int, ::Val{align}) where align =
        @typed_ccall($intr, llvmcall, $T, (LLVMPtr{$T,AS.Global}, Int32), ptr+i-1, align)
end

# interface

export unsafe_cached_load

unsafe_cached_load(p::LLVMPtr{<:Union{LDGTypes...},AS.Global}, i::Integer=1, align::Val=Val(1)) =
    pointerref_ldg(p, Int(i), align)
# NOTE: fall back to normal unsafe_load for unsupported types. we could be smarter here,
#       e.g. destruct/load/reconstruct, but that's too complicated for what it's worth.
unsafe_cached_load(p::LLVMPtr, i::Integer=1, align::Val=Val(1)) =
    unsafe_load(p, i, align)
