# Native execution support

export
    cufunction, @cuda

#
# Auxiliary
#

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

    return args, Tuple{cgtypes...}, Tuple{calltypes...}
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

function compile_function{F<:Function}(func::F, tt)
    @static if TRACE
        # Generate a safe and unique name
        function_uid = "$func-"
        if length(tt.parameters) > 0
            function_uid *= join([replace(string(t), r"\W", "") for t in tt.parameters], '.')
        else
            function_uid *= "Void"
        end

        # Dump the typed AST
        buf = IOBuffer()
        code_warntype(buf, func, tt)
        ast = String(buf)

        output = "$(dumpdir[])/$function_uid.jl"
        trace("Writing kernel AST to $output")
        open(output, "w") do io
            write(io, ast)
        end
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

function emit_allocations(args, codegen_tt, call_tt)
    # if we're generating code for a given type, but passing a pointer to that type instead,
    # this is indicative of needing to upload the value to GPU memory
    kernel_allocations = Expr(:block)
    for i in 1:length(args)
        if call_tt.parameters[i] == Ptr{codegen_tt.parameters[i]}
            @gensym dev_arg
            alloc = quote
                $dev_arg = cualloc($(codegen_tt.parameters[i]))
                # TODO: we're never freeing this
                copy!($dev_arg, $(args[i]))
            end
            append!(kernel_allocations.args, alloc.args)
            args[i] = dev_arg
        end
    end

    return kernel_allocations, args
end

function emit_cudacall(fun, config, args, tt)
    return quote
        cudacall($fun, $config[1], $config[2],
                 $tt, $(args...);
                 shmem_bytes=$config[3])
    end
end


#
# cufunction
#

# Compile and create a CUDA kernel from a Julia function
function cufunction{F<:Function}(func::F, types)
    tt = Base.to_tuple_type(types)
    sig = """$func($(join(tt.parameters, ", ")))"""
    debug("Generating CUDA function for $sig")

    # Check method validity
    ml = Base.methods(func, tt)
    if length(ml) == 0
        error("no method found for kernel $sig")
    elseif length(ml) > 1
        # TODO: when does this happen?
        error("ambiguous call to kernel $sig")
    end
    rt = Base.return_types(func, tt)[1]
    if rt != Void
        error("cannot call kernel $sig as it returns $rt")
    end

    (module_asm, module_entry) = compile_function(func, tt)
    cuda_mod = CuModule(module_asm)
    cuda_fun = CuFunction(cuda_mod, module_entry)

    return cuda_fun, cuda_mod
end


#
# @cuda macro
#

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

    esc(:(CUDAnative.generated_cuda($config, $(callexpr.args...))))
end

# TODO: generated_cuda behaves subtly different when passing a CuFunction (no use of
#       cgtypes, changes behaviour wrt. ghost types and type conversions). Iron this out,
#       and create tests.

# Execute a pre-compiled CUDA kernel
@generated function generated_cuda(config::Tuple{CuDim, CuDim, Int},
                                      cuda_fun::CuFunction, argspec...)
    args, _, call_tt = decode_arguments(argspec)
    any(t->t==Base.Bottom, call_tt.parameters) && error("cannot pass non-concrete types to precompiled kernels")

    kernel_allocations, args = emit_allocations(args, call_tt, call_tt)

    kernel_call = emit_cudacall(:(cuda_fun), :(config), args, call_tt)

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, kernel_allocations.args)
    append!(exprs.args, kernel_call.args)
    return exprs
end

# Compile and execute a CUDA kernel from a Julia function
const func_cache = Dict{UInt, CuFunction}()
@generated function generated_cuda{F<:Function}(config::Tuple{CuDim, CuDim, Int},
                                                   func::F, argspec...)
    args, codegen_tt, call_tt = decode_arguments(argspec)

    kernel_allocations, args = emit_allocations(args, codegen_tt, call_tt)

    @gensym cuda_fun
    key = hash(Base.tt_cons(func, codegen_tt))
    kernel_compilation = quote
        if (haskey(CUDAnative.func_cache, $key))
            $cuda_fun = CUDAnative.func_cache[$key]
        else
            $cuda_fun, _ = cufunction(func, $codegen_tt)
            CUDAnative.func_cache[$key] = $cuda_fun
        end
    end

    # filter out non-concrete args
    concrete = map(t->t!=Base.Bottom, call_tt.parameters)
    concrete_call_tt = Tuple{map(x->x[2], filter(x->x[1], zip(concrete, call_tt.parameters)))...}
    concrete_args    =       map(x->x[2], filter(x->x[1], zip(concrete, args)))

    kernel_call = emit_cudacall(cuda_fun, :(config), concrete_args, concrete_call_tt)

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, kernel_allocations.args)
    append!(exprs.args, kernel_compilation.args)
    append!(exprs.args, kernel_call.args)
    return exprs
end
