# Native execution support

export
    @cuda

# Convenience macro to support a naturally looking call
# eg. @cuda (len, 1) kernel_foo(bar)
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
    if argtype <: Ptr
        error("cannot pass host pointers to a device function")
    elseif argtype <: DevicePtr
        # known convert to raw pointers
        cgtype = argtype
        calltype = argtype
    elseif argtype.layout != C_NULL && Base.datatype_pointerfree(argtype)
        # pointerfree objects with a layout can be used on the GPU
        cgtype = argtype
        # but the ABI might require them to be passed by pointer
        if sizeof(argtype) > 8  # TODO: verify this at the LLVM side
            calltype = DevicePtr{argtype}
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

function compile_function{F<:Function}(ftype::Type{F}, types::Vector{DataType})
    debug("Compiling $ftype$types")

    # Try to get a hold of the original function
    # NOTE: this doesn't work for closures, as there can be multiple instances
    #       with each a different environment
    f = Nullable{Function}()
    try
        f = Nullable{Function}(ftype.instance)
    end

    @static if TRACE
        # Generate a safe and unique name
        function_uid = "$(get(f))-"
        if length(types) > 0
            function_uid *= join([replace(string(x), r"\W", "")
                                for x in types], '.')
        else
            function_uid *= "Void"
        end

        # Dump the typed AST
        if !isnull(f)
            buf = IOBuffer()
            code_warntype(buf, get(f), types)
            ast = String(buf)

            output = "$(dumpdir[])/$function_uid.jl"
            trace("Writing kernel AST to $output")
            open(output, "w") do io
                write(io, ast)
            end
        end
    end

    # Generate LLVM IR
    # NOTE: this is lifted from reflection.jl::_dump_function
    t = Base.tt_cons(ftype, Base.to_tuple_type(types))
    # Next call will trigger compilation (if necessary)
    llvmf = ccall(:jl_get_llvmf, Ptr{Void}, (Any, Bool, Bool), t, false, true)
    if llvmf == C_NULL
        error("no method found for kernel $ftype$types")
    end

    # Generate (PTX) assembly
    module_asm = ccall(:jl_dump_function_asm, Any, (Ptr{Void},Cint), llvmf, 0)::String

    # Get entry point
    module_entry = ccall(:jl_dump_function_name, Any, (Ptr{Void},), llvmf)::String

    trace("Function entry point: $module_entry")

    # Dump the LLVM IR
    @static if TRACE
        if !isnull(f)
            buf = IOBuffer()
            code_llvm(buf, get(f), types, #=strip_di=#false, #=entire_module=#true)
            module_llvm = String(buf)

            output = "$(dumpdir[])/$function_uid.ll"
            trace("Writing kernel LLVM IR to $output")
            open(output, "w") do io
                write(io, module_llvm)
            end
        end
    end

    # Dump the PTX assembly
    @static if TRACE
        output = "$(dumpdir[])/$function_uid.ptx"
        trace("Writing kernel PTX assembly to $output")
        open(output, "w") do io
            write(io, module_asm)
        end
    end

    # Return module & module entry containing function
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
            args[i] = :( convert($ct, $(args[i])) )
        end
    end

    # figure out how to codegen and pass these types
    cgtypes, calltypes = Array{DataType}(length(argtypes)), Array{Type}(length(argtypes))
    for i in 1:length(args)
        cgtypes[i], calltypes[i] = physical_type(argtypes[i])
    end

    return args, cgtypes, calltypes
end

function create_allocations(args, cgtypes, calltypes)
    # if we're generating code for a given type, but passing a pointer to that type instead,
    # this is indicative of needing to upload the value to GPU memory
    kernel_allocations = Expr(:block)
    for i in 1:length(args)
        if calltypes[i] == DevicePtr{cgtypes[i]}
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

function create_call(cuda_fun, config, args, calltypes)
    return :( cudacall($cuda_fun, $config[1], $config[2],
                       $(tuple(calltypes...)), $(args...);
                       shmem_bytes=$config[3]) )
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
    push!(exprs.args, kernel_call)
    return exprs
end

# Compile a kernel and generate a cudacall
const func_cache = Dict{UInt, CuFunction}()
@generated function generate_cudacall{F<:Function}(config::Tuple{CuDim, CuDim, Int},
                                                   ftype::F, argspec...)
    args, cgtypes, calltypes = decode_arguments(argspec)

    kernel_allocations, args = create_allocations(args, cgtypes, calltypes)

    # filter out non-concrete args
    concrete = map(t->t!=Base.Bottom, calltypes)
    concrete_calltypes = map(x->x[2], filter(x->x[1], zip(concrete, calltypes)))
    concrete_args      = map(x->x[2], filter(x->x[1], zip(concrete, args)))

    @gensym cuda_fun
    key = hash((ftype, cgtypes...))
    kernel_compilation = quote
        if (haskey(CUDAnative.func_cache, $key))
            $cuda_fun = CUDAnative.func_cache[$key]
        else
            (module_asm, module_entry) = CUDAnative.compile_function($ftype, $cgtypes)
            cuda_mod = CuModuleData(module_asm)
            $cuda_fun = CuFunction(cuda_mod, module_entry)
            CUDAnative.func_cache[$key] = $cuda_fun
        end
    end

    kernel_call = create_call(cuda_fun, :(config), concrete_args, concrete_calltypes)

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, kernel_allocations.args)
    append!(exprs.args, kernel_compilation.args)
    push!(exprs.args, kernel_call)
    return exprs
end
