# specialization facts: launch-checked bounds on kernel arguments
#
# The dimensions of a `CuDeviceArray` are `Int`s, so index computations involving them are
# performed using 64-bit integers, which GPUs execute as pairs of 32-bit instructions. Almost
# every array is small enough to be indexed using 32-bit integers, but that is a property of
# the values, not of the types the kernel is compiled for.
#
# So we check the values when launching a kernel: if every extent (the dimensions and length
# of a device array, or the stop of a `OneTo`, as found in `CartesianIndices` and KA's
# iteration spaces) of the converted arguments fits in an `Int32`, the kernel is compiled with
# `index_facts=true`. The compiler then emits `llvm.assume(extent <= typemax(Int32))` on the
# kernel's parameters, which the narrowing pass (see `narrowing.jl`) uses to perform index
# computations using 32-bit integers.
#
# The facts are only attached to the parameter values, so arrays created on the device (e.g.,
# using `reshape` or shared memory) don't inherit them. Every launch of a kernel compiled with
# facts checks its arguments again, and falls back to a kernel compiled without facts if they
# don't fit (see `kernel_launch`).

# the paths (as sequences of field indices) to the extents of a value of type `T`
function extent_paths(@nospecialize(T))
    paths = Vector{Int}[]
    extent_paths!(paths, Int[], T)
    return paths
end
function extent_paths!(paths, path, @nospecialize(T))
    (T isa DataType && isbitstype(T) && sizeof(T) > 0) || return
    if T <: CuDeviceArray
        dims = Base.fieldindex(T, :dims)
        for i in 1:ndims(T)
            push!(paths, [path..., dims, i])
        end
        push!(paths, [path..., Base.fieldindex(T, :len)])
    elseif T === Base.OneTo{Int}
        push!(paths, [path..., Base.fieldindex(T, :stop)])
    elseif T <: Ptr || T <: Core.LLVMPtr || isprimitivetype(T)
        # no extents
    else
        for i in 1:fieldcount(T)
            extent_paths!(paths, [path..., i], fieldtype(T, i))
        end
    end
    return
end

# the byte offset of an extent within a value of type `T`
function extent_offset(@nospecialize(T), path)
    offset = 0
    for i in path
        offset += fieldoffset(T, i)
        T = fieldtype(T, i)
    end
    @assert T === Int
    return Int(offset)
end

# do the extents of `x` fit in an `Int32`? generated from the same paths as the facts that
# the compiler emits, so that they cannot diverge, and unrolled to straight-line code.
@inline @generated function extents_fit(x)
    checks = map(extent_paths(x)) do path
        ex = :x
        for i in path
            ex = :(getfield($ex, $i))
        end
        :($ex % UInt <= $(UInt(typemax(Int32))))
    end
    isempty(checks) ? true : Expr(:call, :&, checks...)
end

# do the extents of all arguments fit? arguments of a different type than the ones the
# kernel was compiled for (which are converted when launching) are conservatively rejected.
@inline @generated function arguments_fit(::Type{TT}, args::Tuple) where {TT}
    checks = map(1:fieldcount(args)) do i
        if fieldtype(args, i) === fieldtype(TT, i)
            :(extents_fit(args[$i]))
        else
            false
        end
    end
    isempty(checks) ? true : Expr(:call, :&, checks...)
end

# emit the facts at the start of a kernel. this happens before the parameters are lowered
# to values, so the extents are loaded from the parameters passed by reference.
function emit_index_facts!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                           entry::LLVM.Function)
    args = GPUCompiler.classify_arguments(job, entry.function_type)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()
    bound = ConstantInt(T_i64, typemax(Int32))
    assume = LLVM.Function(mod, LLVM.Intrinsic("llvm.assume"))
    @dispose builder=IRBuilder() begin
        position!(builder, LLVM.before(first(first(entry.blocks).instructions)))
        for arg in args
            arg.cc == GPUCompiler.BITS_REF || continue
            param = entry.parameters[arg.idx]
            for path in extent_paths(arg.typ)
                offset = ConstantInt(T_i64, extent_offset(arg.typ, path))
                ptr = inbounds_gep!(builder, T_i8, param, [offset])
                val = load!(builder, T_i64, ptr)
                cond = icmp!(builder, LLVM.API.LLVMIntULE, val, bound)
                call!(builder, assume.function_type, assume, [cond])
            end
        end
    end
    return
end
