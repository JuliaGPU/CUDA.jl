const MATH_INTRINSICS = Dict("llvm.$op" => op for op in
    (:exp, :exp2, :exp10, :log, :log2, :log10, :sin, :cos, :sincos,
     :tan, :asin, :acos, :atan, :atan2, :sinh, :cosh, :tanh, :pow, :powi))

const FAST_LIBDEVICE_MATH = (:exp, :exp10, :log, :log2, :log10, :sin, :cos,
                            :sincos, :tan, :pow)

function half_math_call!(builder, op, typ, args)
    name = "__cuda_$(op)_$(GPUCompiler.type_suffix(typ))"
    if typ isa LLVM.BFloatType && !BFloat16s.llvm_arithmetic
        integer = LLVM.Int16Type()
        bits = LLVM.Value[bitcast!(builder, only(args), integer)]
        result = GPUCompiler.call_declared!(builder, name, integer, bits)
        return bitcast!(builder, result, typ)
    end
    GPUCompiler.call_declared!(builder, name, typ, args)
end

function libdevice_math_call!(builder, op, typ, args, fast)
    name = "__nv_"
    fast && typ isa LLVM.FloatType && op in FAST_LIBDEVICE_MATH && (name *= "fast_")
    name *= string(op)
    typ isa LLVM.FloatType && (name *= "f")
    if op === :sincos
        sine, cosine = @dispose allocations=IRBuilder() begin
            entry = first(builder.insert_block.parent.blocks)
            position!(allocations, LLVM.before(first(entry.instructions)))
            (alloca!(allocations, typ), alloca!(allocations, typ))
        end
        value = GPUCompiler.call_declared!(builder, name, LLVM.VoidType(),
                                           LLVM.Value[only(args), sine, cosine])
        result_type = LLVM.StructType(LLVMType[typ, typ])
        result = insert_value!(builder, PoisonValue(result_type), load!(builder, typ, sine), 0)
        result = insert_value!(builder, result, load!(builder, typ, cosine), 1)
    else
        value = GPUCompiler.call_declared!(builder, name, typ, args)
        result = value
    end
    fun = value.called_function
    push!(fun.function_attributes, EnumAttribute("nounwind"))
    push!(fun.function_attributes, EnumAttribute("willreturn"))
    fun.memory_effects = op === :sincos ? LLVM.MemoryEffects(; argmem=:write) :
                                        LLVM.MemoryEffects()
    return result
end

function lower_math_call!(job, builder, call, op)
    args = LLVM.Value[call.arguments...]
    typ = first(args).value_type
    typ isa Union{LLVM.HalfType,LLVM.BFloatType,LLVM.FloatType,LLVM.DoubleType} ||
        error("Unsupported CUDA math intrinsic type $typ")
    fast = call.fast_math.afn || job.config.target.fastmath
    target = job.config.target
    if fast && typ isa LLVM.HalfType && op in (:exp2, :tanh) &&
       target.cap >= v"7.5" && target.ptx >= v"7.0"
        call.fast_math = merge(NamedTuple(call.fast_math), (; afn=true))
        return nothing
    end
    if op === :tanh && fast && typ isa LLVM.FloatType &&
       target.cap >= v"7.5" && target.ptx >= v"7.0"
        call.fast_math = merge(NamedTuple(call.fast_math), (; afn=true))
        return nothing
    end
    if op === :powi
        exponent = args[2]
        width = exponent.value_type.width
        if width < 32
            args[2] = sext!(builder, exponent, LLVM.Int32Type())
        elseif width > 32
            exponent isa ConstantInt && typemin(Int32) <= convert(BigInt, exponent) <= typemax(Int32) ||
                error("CUDA llvm.powi requires an exponent representable as Int32")
            args[2] = trunc!(builder, exponent, LLVM.Int32Type())
        end
    end
    if typ isa Union{LLVM.HalfType,LLVM.BFloatType}
        if op === :sincos
            result = PoisonValue(call.value_type)
            for (index, component) in enumerate((:sin, :cos))
                value = half_math_call!(builder, component, typ, args)
                result = insert_value!(builder, result, value, index - 1)
            end
            return result
        end
        if op in (:exp, :exp2, :exp10, :log, :log2, :log10, :sin, :cos) ||
           (typ isa LLVM.BFloatType && op === :tanh)
            return half_math_call!(builder, op, typ, args)
        end
        promoted = LLVM.Value[arg.value_type == typ ? fpext!(builder, arg, LLVM.FloatType()) :
                                                     arg for arg in args]
        value = libdevice_math_call!(builder, op, LLVM.FloatType(), promoted, fast)
        return fptrunc!(builder, value, typ)
    end
    libdevice_math_call!(builder, op, typ, args, fast)
end

function lower_math_intrinsics!(job, mod)
    table = GPUCompiler.intrinsic_table(MATH_INTRINSICS)
    for fun in collect(mod.functions)
        isdeclaration(fun) && continue
        GPUCompiler.scalarize_vector_calls!((call, intr) -> haskey(table, intr), fun)
        GPUCompiler.lower_intrinsic_calls!(fun) do builder, call, intrinsic
            op = get(table, intrinsic, nothing)
            op === nothing && return nothing
            lower_math_call!(job, builder, call, op)
        end
    end
    return
end

function GPUCompiler.lower_intrinsics!(@nospecialize(job::AnyCUDAJob), mod::LLVM.Module,
                                       entry::LLVM.Function)
    lower_math_intrinsics!(job, mod)
    entry
end

function GPUCompiler.validate_ir(job::AnyCUDAJob, mod::LLVM.Module)
    errors = invoke(GPUCompiler.validate_ir,
                    Tuple{CompilerJob{PTXCompilerTarget},LLVM.Module}, job, mod)
    job.config.libraries || return errors
    table = GPUCompiler.intrinsic_table(MATH_INTRINSICS)
    for fun in mod.functions, block in fun.blocks, inst in block.instructions
        op = get(table, GPUCompiler.called_intrinsic(inst), nothing)
        op === nothing && continue
        if op === :tanh && inst.value_type isa LLVM.FloatType && inst.fast_math.afn &&
           job.config.target.cap >= v"7.5" && job.config.target.ptx >= v"7.0"
            continue
        end
        if op in (:exp2, :tanh) && inst.value_type isa LLVM.HalfType &&
           inst.fast_math.afn && job.config.target.cap >= v"7.5" &&
           job.config.target.ptx >= v"7.0"
            continue
        end
        push!(errors, ("unlowered CUDA math intrinsic", GPUCompiler.backtrace(inst), string(inst)))
    end
    errors
end
