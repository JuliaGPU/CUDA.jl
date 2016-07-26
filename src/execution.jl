# Native execution support

export
    @cuda

# Convenience macro to support a naturally looking call
# eg. @cuda (1,len) kernel_foo(bar)
macro cuda(config::Expr, callexpr::Expr)
    # Sanity checks
    if config.head != :tuple || !(2 <= length(config.args) <= 3)
        throw(ArgumentError("first argument to @cuda should be a tuple (gridDim, blockDim, [shmem])"))
    end
    if length(config.args) == 2
        push!(config.args, :0)
    end
    if callexpr.head != :call
        throw(ArgumentError("second argument to @cuda should be a function call"))
    end

    esc(:(CUDAnative.generate_cudacall($config, $(callexpr.args...))))
end

# Determine the physical type of an object, that is, the types that will be used to
# specialize the kernel function, and to convert argument values to.
#
# These two types can differ, eg. when passing a bitstype that doesn't fit in a register, in
# which case we'll be specializing a function that directly uses that bitstype (and the
# codegen ABI will figure out it needs to be passed by ref, ie. a pointer), but we do need
# to allocate memory and pass an actual pointer since we perform the call ourselves.
function physical_type(argtype::DataType)
    if argtype.layout != C_NULL && Base.datatype_pointerfree(argtype)
        # pointerfree objects with a layout can be used on the GPU
        cgtype = argtype
        # but the ABI might require them to be passed by pointer
        if sizeof(argtype) > 8  # TODO: verify this at the LLVM side
            calltype = Ptr{argtype}
        else
            calltype = argtype
        end
    else
        error("don't know how to handle argument of type $argtype")
    end

    # special-case args which don't appear in the generated code
    # (but we still need to specialize for)
    if !argtype.mutable && sizeof(argtype) == 0
        # ghost type, ignored by the compiler
        calltype = Base.Bottom
    end

    return cgtype::Type, calltype::Type
end

function compile_function{F<:Function}(func::F, types::Vector{DataType})    # TODO: make types tuple?
    debug("Compiling $func($(types...))")

    @static if TRACE
        # Generate a safe and unique name
        function_uid = "$func-"
        if length(types) > 0
            function_uid *= join([replace(string(x), r"\W", "") for x in types], '.')
        else
            function_uid *= "Void"
        end

        # Dump the typed AST
        buf = IOBuffer()
        code_warntype(buf, func, types)
        ast = String(buf)

        output = "$(dumpdir[])/$function_uid.jl"
        trace("Writing kernel AST to $output")
        open(output, "w") do io
            write(io, ast)
        end
    end

    # Check method validity
    tt = tuple(types...)
    ml = Base.methods(func, tt)
    if length(ml) == 0
        error("no method found for kernel $func($(types...))")
    elseif length(ml) > 1
        # TODO: when does this happen?
        error("ambiguous call to kernel $func($(types...))")
    end
    rt = Base.return_types(func, tt)[1]
    if rt != Void
        error("cannot call $func($(types...)) with @cuda as it returns $rt")
    end

    # Generate LLVM IR
    @static if TRACE
        module_llvm = sprint(io->code_llvm(io, func, tt, #=strip_di=#false, #=entire_module=#true))

        output = "$(dumpdir[])/$function_uid.ll"
        trace("Writing kernel LLVM IR to $output")
        open(output, "w") do io
            write(io, module_llvm)
        end
    end

    # Generate (PTX) assembly
    module_asm = sprint(io->code_native(io, func, tt))
    @static if TRACE
        output = "$(dumpdir[])/$function_uid.ptx"
        trace("Writing kernel PTX assembly to $output")
        open(output, "w") do io
            write(io, module_asm)
        end
    end

    # Get entry point
    # TODO: get rid of the llvmf (lifted from reflection.jl)
    llvmf = ccall(:jl_get_llvmf, Ptr{Void},
                 (Any, Bool, Bool),
                 Base.tt_cons(typeof(func), tt), false, true)
    module_entry = ccall(:jl_dump_function_name, Any, (Ptr{Void},), llvmf)::String
    trace("Function entry point: $module_entry")

    return (module_asm, module_entry)
end

function decode_arguments(argspec)
    argtypes = DataType[argspec...]
    args = Union{Expr,Symbol}[:( argspec[$i] ) for i in 1:length(argtypes)]

    # convert types to their CUDA representation
    for i in 1:length(args)
        t = argtypes[i]
        ct = cudaconvert(t)
        if ct != t
            argtypes[i] = ct
            if ct <: Ptr
                args[i] = :( unsafe_convert($ct, $(args[i])) )
            else
                args[i] = :( convert($ct, $(args[i])) )
            end
        end
    end

    # figure out how to codegen and pass these types
    cgtypes, calltypes = Array{DataType}(length(argtypes)), Array{Type}(length(argtypes))
    for i in 1:length(args)
        cgtypes[i], calltypes[i] = physical_type(argtypes[i])
    end

    # NOTE: DevicePtr's should have disappeared after this point

    return args, cgtypes, calltypes
end

function create_allocations(args, cgtypes, calltypes)
    # if we're generating code for a given type, but passing a pointer to that type instead,
    # this is indicative of needing to upload the value to GPU memory
    kernel_allocations = Expr(:block)
    for i in 1:length(args)
        if calltypes[i] == Ptr{cgtypes[i]}
            @gensym dev_arg
            alloc = quote
                $dev_arg = cualloc($(cgtypes[i]))
                # TODO: we're never freeing this
                copy!($dev_arg, $(args[i]))
            end
            append!(kernel_allocations.args, alloc.args)
            args[i] = dev_arg
        end
    end

    return kernel_allocations, args
end

function create_call(fun, config, args, calltypes)
    return quote
        cudacall($fun, $config[1], $config[2],
                 $(tuple(calltypes...)), $(args...);
                 shmem_bytes=$config[3])
    end
end

# TODO: generate_cudacall behaves subtly different when passing a CuFunction (no use of
#       cgtypes, changes behaviour wrt. ghost types and type conversions). Iron this out,
#       and create tests.

# Generate a cudacall to a pre-compiled CUDA function
@generated function generate_cudacall(config::Tuple{CuDim, CuDim, Int},
                                      cuda_fun::CuFunction, argspec...)
    args, _, calltypes = decode_arguments(argspec)
    any(t->t==Base.Bottom, calltypes) && error("cannot pass non-concrete types to precompiled kernels")

    kernel_allocations, args = create_allocations(args, calltypes, calltypes)

    kernel_call = create_call(:(cuda_fun), :(config), args, calltypes)

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, kernel_allocations.args)
    append!(exprs.args, kernel_call.args)
    return exprs
end

# Compile a kernel and generate a cudacall
const func_cache = Dict{UInt, CuFunction}()
@generated function generate_cudacall{F<:Function}(config::Tuple{CuDim, CuDim, Int},
                                                   func::F, argspec...)
    args, cgtypes, calltypes = decode_arguments(argspec)

    kernel_allocations, args = create_allocations(args, cgtypes, calltypes)

    # filter out non-concrete args
    concrete = map(t->t!=Base.Bottom, calltypes)
    concrete_calltypes = map(x->x[2], filter(x->x[1], zip(concrete, calltypes)))
    concrete_args      = map(x->x[2], filter(x->x[1], zip(concrete, args)))

    @gensym cuda_fun
    key = hash((func, cgtypes...))
    kernel_compilation = quote
        if (haskey(CUDAnative.func_cache, $key))
            $cuda_fun = CUDAnative.func_cache[$key]
        else
            (module_asm, module_entry) = CUDAnative.compile_function(func, $cgtypes)
            cuda_mod = CuModule(module_asm)
            $cuda_fun = CuFunction(cuda_mod, module_entry)
            CUDAnative.func_cache[$key] = $cuda_fun
        end
    end

    kernel_call = create_call(cuda_fun, :(config), concrete_args, concrete_calltypes)

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, kernel_allocations.args)
    append!(exprs.args, kernel_compilation.args)
    append!(exprs.args, kernel_call.args)
    return exprs
end
