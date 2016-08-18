# Native execution support

export
    cufunction, @cuda

#
# Auxiliary
#

"""
Convert the arguments to a kernel function to their CUDA representation, and figure out what
types to specialize the kernel function for and how to actually pass those objects.
"""
function convert_arguments(args, tt)
    argtypes = DataType[tt.parameters...]
    argexprs = Union{Expr,Symbol}[args...]

    # convert types to their CUDA representation
    for i in 1:length(argexprs)
        t = argtypes[i]
        ct = cudaconvert(t)
        if ct != t
            argtypes[i] = ct
            if ct <: Ptr
                argexprs[i] = :( unsafe_convert($ct, $(argexprs[i])) )
            else
                argexprs[i] = :( convert($ct, $(argexprs[i])) )
            end
        end
    end

    # figure out how to codegen and pass these types
    cgtypes, calltypes = Array{DataType}(length(argtypes)), Array{Type}(length(argtypes))
    for i in 1:length(argexprs)
        cgtypes[i], calltypes[i] = actual_types(argtypes[i])
    end

    # NOTE: DevicePtr's should have disappeared after this point

    return argexprs, Tuple{cgtypes...}, Tuple{calltypes...}
end

"""
Determine the actual types of an object, that is, 1) the type that needs to be used to
specialize (compile) the kernel function, and 2) the type which an object needs to be
converted to before passing it to a kernel.

These two types can differ, eg. when passing a bitstype that doesn't fit in a register, in
which case we'll be specializing a function that directly uses that bitstype (and the
codegen ABI will figure out it needs to be passed by ref, ie. a pointer), but we do need to
allocate memory and pass an actual pointer since we perform the call ourselves.
"""
function actual_types(argtype::DataType)
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

"""
Compile a function to PTX, returning the assembly and an entry point.
Not to be used directly, see `cufunction` instead.
"""
function compile_function{F<:Function}(func::F, tt)
    @static if TRACE
        # generate a safe and unique name
        function_uid = "$func-"
        if length(tt.parameters) > 0
            function_uid *= join([replace(string(t), r"\W", "") for t in tt.parameters], '.')
        else
            function_uid *= "Void"
        end

        # dump the typed AST
        buf = IOBuffer()
        code_warntype(buf, func, tt)
        ast = String(buf)

        output = "$(dumpdir[])/$function_uid.jl"
        trace("Writing kernel AST to $output")
        open(output, "w") do io
            write(io, ast)
        end
    end

    # generate LLVM IR
    @static if TRACE
        module_llvm = sprint(io->code_llvm(io, func, tt, #=strip_di=#false, #=entire_module=#true))

        output = "$(dumpdir[])/$function_uid.ll"
        trace("Writing kernel LLVM IR to $output")
        open(output, "w") do io
            write(io, module_llvm)
        end
    end

    # generate (PTX) assembly
    module_asm = sprint(io->code_native(io, func, tt))
    @static if TRACE
        output = "$(dumpdir[])/$function_uid.ptx"
        trace("Writing kernel PTX assembly to $output")
        open(output, "w") do io
            write(io, module_asm)
        end
    end

    # extract entry point
    module_entry = Nullable{String}()
    entry_re = r"^\.visible \.entry (\w+)\("
    buf = IOBuffer(module_asm)
    for ln in eachline(buf)
        m = match(entry_re, ln)
        if m != nothing
            isnull(module_entry) || error("multiple entry-points functions in module")
            module_entry = Nullable{String}(m.captures[1])
        end
    end
    isnull(module_entry) && error("no entry-points functions in module")
    trace("Function entry point: ", get(module_entry))

    return module_asm, get(module_entry)
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

function emit_cudacall(func, dims, kwargs, args, tt::Type)
    # TODO: can we handle non-isbits types? 
    #       if so, move this more stringent check to @cuda(CuFunction)
    all(t -> isbits(t) && sizeof(t) > 0, tt.parameters) ||
        error("can only pass bitstypes of size > 0 to CUDA kernels")
    any(t -> sizeof(t) > 8, tt.parameters) &&
        error("cannot pass objects that don't fit in registers to CUDA functions")

    return quote
        cudacall($func, $dims[1], $dims[2], $tt, $(args...); $kwargs...)
    end
end


#
# cufunction
#

# Compile and create a CUDA function from a Julia function
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

"""
    @cuda (gridDim::CuDim, blockDim::CuDim, [shmem::Int], [stream::CuStream]) func(args...)

High-level interface for calling functions on a GPU. The `gridDim` and `blockDim` arguments
represent the launch configuration, the optional `shmem` parameter specifies how much bytes
of dynamic shared memory should be allocated (defaults to 0).

When `func` represents a `CuFunction` object, a `cudacall` will be emitted, with the type
signature derived from the actual argument types (if you need conversions to happen, use
`cudacall` directly). Only bitstypes with size ∈ ]0,8] are supported (0-sized elements are
ghost types that wouldn't get passed, while objects > 8 bytes don't fit in registers and
need to be passed by pointer instead).

If `func` represents a Julia function, a more automagic approach is taken. The function is
compiled to a CUDA function, more objects are supported (ghost types are used during codegen
but not passed, objects > 8 bytes are copied to memory and passed by pointer), and finally a
`cudacall` is performed.
"""
macro cuda(config::Expr, callexpr::Expr)
    # sanity checks
    if config.head != :tuple || !(2 <= length(config.args) <= 4)
        throw(ArgumentError("first argument to @cuda should be a tuple (gridDim, blockDim, [shmem], [stream])"))
    end
    if callexpr.head != :call
        throw(ArgumentError("second argument to @cuda should be a function call"))
    end

    # optional tuple elements are forwarded to `cudacall` by means of kwargs
    if length(config.args) == 2
        return esc(:(CUDAnative.generated_cuda($config[1:2], $(callexpr.args...))))
    elseif length(config.args) == 3
        return esc(:(CUDAnative.generated_cuda($config[1:2], $(callexpr.args...);
                                               shmem=$config[3])))
    elseif length(config.args) == 4
        return esc(:(CUDAnative.generated_cuda($config[1:2], $(callexpr.args...);
                                               shmem=$config[3], stream=$config[4])))
    end
end

# Execute a pre-compiled CUDA kernel
@generated function generated_cuda(dims::Tuple{CuDim, CuDim},
                                   cuda_fun::CuFunction, argspec...;
                                   kwargs...)
    tt = Base.to_tuple_type(argspec)
    args = [:( argspec[$i] ) for i in 1:length(argspec)]

    return emit_cudacall(:(cuda_fun), :(dims), :(kwargs), args, tt)
end

# Compile and execute a CUDA kernel from a Julia function
const func_cache = Dict{UInt, CuFunction}()
@generated function generated_cuda{F<:Function}(dims::Tuple{CuDim, CuDim},
                                                func::F, argspec...;
                                                kwargs...)
    tt = Base.to_tuple_type(argspec)
    args = [:( argspec[$i] ) for i in 1:length(argspec)]
    args, codegen_tt, call_tt = convert_arguments(args, tt)

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

    kernel_call = emit_cudacall(cuda_fun, :(dims), :(kwargs), concrete_args, concrete_call_tt)

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, kernel_allocations.args)
    append!(exprs.args, kernel_compilation.args)
    append!(exprs.args, kernel_call.args)
    return exprs
end
