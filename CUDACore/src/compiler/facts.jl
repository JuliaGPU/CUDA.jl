# specialization facts: launch-checked bounds on kernel arguments
#
# The dimensions of a `CuDeviceArray` are `Int`s, so index computations involving them are
# performed using 64-bit integers, which GPUs execute as pairs of 32-bit instructions. Almost
# every array is small enough to be indexed using 32-bit integers, but that is a property of
# the values, not of the types the kernel is compiled for.
#
# So we check the values when launching a kernel: if every extent (the dimensions and length
# of a device array, or the stop of a `OneTo`, as found in `CartesianIndices` and KA's
# iteration spaces) of the converted arguments is in `0:typemax(Int32)`, and every other
# `Int` (offsets, ranges, sizes and scalar arguments that enter index computations) fits in
# an `Int32`, the kernel is compiled with `index_facts=true`. The compiler then emits
# `llvm.assume`s stating these bounds on the kernel's parameters, which the narrowing pass
# (see `narrowing.jl`) uses to perform index computations using 32-bit integers.
#
# The facts are only attached to the parameter values, so arrays created on the device (e.g.,
# using `reshape` or shared memory) don't inherit them. Every launch of a kernel compiled with
# facts checks its arguments again, and falls back to a kernel compiled without facts if they
# don't fit (see `kernel_launch`).

# the `Int`s of a value of type `T` that facts bound, as paths (sequences of field indices)
# and whether the bound is signed: extents are in `0:typemax(Int32)`, other `Int`s in the
# range of an `Int32`
function fact_paths(@nospecialize(T))
    paths = Pair{Vector{Int},Bool}[]
    fact_paths!(paths, Int[], T)
    return paths
end
function fact_paths!(paths, path, @nospecialize(T))
    (T isa DataType && isbitstype(T) && sizeof(T) > 0) || return
    if T <: CuDeviceArray
        dims = Base.fieldindex(T, :dims)
        for i in 1:ndims(T)
            push!(paths, [path..., dims, i] => false)
        end
        push!(paths, [path..., Base.fieldindex(T, :len)] => false)
        # not `maxsize`: a size in bytes, which may exceed `typemax(Int32)` for arrays that
        # can still be indexed using 32-bit integers
    elseif T === Base.OneTo{Int}
        push!(paths, [path..., Base.fieldindex(T, :stop)] => false)
    elseif T === Int
        push!(paths, copy(path) => true)
    elseif T <: Ptr || T <: Core.LLVMPtr || isprimitivetype(T)
        # no indices
    else
        for i in 1:fieldcount(T)
            fact_paths!(paths, [path..., i], fieldtype(T, i))
        end
    end
    return
end

# the byte offset of an `Int` within a value of type `T`
function fact_offset(@nospecialize(T), path)
    offset = 0
    for i in path
        offset += fieldoffset(T, i)
        T = fieldtype(T, i)
    end
    @assert T === Int
    return Int(offset)
end

# do the `Int`s of `x` satisfy the facts? generated from the same paths as the facts that
# the compiler emits, so that they cannot diverge, and unrolled to straight-line code.
@inline @generated function values_fit(x)
    checks = map(fact_paths(x)) do (path, signed)
        ex = :x
        for i in path
            ex = :(getfield($ex, $i))
        end
        if signed
            :(($ex - $(Int(typemin(Int32)))) % UInt <= $(UInt(typemax(UInt32))))
        else
            :($ex % UInt <= $(UInt(typemax(Int32))))
        end
    end
    isempty(checks) ? true : Expr(:call, :&, checks...)
end

# do all arguments satisfy the facts? arguments of a different type than the ones the
# kernel was compiled for (which are converted when launching) are conservatively rejected.
@inline @generated function arguments_fit(::Type{TT}, args::Tuple) where {TT}
    checks = map(1:fieldcount(args)) do i
        if fieldtype(args, i) === fieldtype(TT, i)
            :(values_fit(args[$i]))
        else
            false
        end
    end
    isempty(checks) ? true : Expr(:call, :&, checks...)
end

# emit the facts at the start of a kernel. this happens before the parameters are lowered
# to values, so the fields of aggregates are loaded from the parameters passed by reference.
function emit_index_facts!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                           entry::LLVM.Function)
    args = GPUCompiler.classify_arguments(job, entry.function_type)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()
    @dispose builder=IRBuilder() begin
        position!(builder, LLVM.before(first(first(entry.blocks).instructions)))
        function bound!(val, signed)
            if signed
                lo = icmp!(builder, LLVM.API.LLVMIntSGE, val,
                           ConstantInt(T_i64, typemin(Int32)))
                assume!(builder, lo)
            end
            hi = icmp!(builder, signed ? LLVM.API.LLVMIntSLE : LLVM.API.LLVMIntULE, val,
                       ConstantInt(T_i64, typemax(Int32)))
            assume!(builder, hi)
        end
        for arg in args
            if arg.cc == GPUCompiler.BITS_VALUE && arg.typ === Int
                bound!(entry.parameters[arg.idx], true)
            elseif arg.cc == GPUCompiler.BITS_REF
                param = entry.parameters[arg.idx]
                for (path, signed) in fact_paths(arg.typ)
                    offset = ConstantInt(T_i64, fact_offset(arg.typ, path))
                    ptr = inbounds_gep!(builder, T_i8, param, [offset])
                    bound!(load!(builder, T_i64, ptr), signed)
                end
            end
        end
    end
    return
end
