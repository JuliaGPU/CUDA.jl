# Native execution support

export
    @cuda


#
# macros/functions for native Julia-CUDA processing
#

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

    esc(:(CUDAnative.generate_launch($config, $(callexpr.args...))))
end

immutable ArgRef
    typ::Type
    ref::Union{Symbol, Expr}
end

# Read the type and symbol information for each argument of the generated
# function
function read_arguments(argspec::Tuple)
    args = Array(ArgRef, length(argspec))

    for i in 1:length(argspec)
        args[i] = ArgRef(argspec[i], :( argspec[$i] ))

        if args[i].typ <: DevicePtr{Void} || isbits(args[i].typ)
            # these will be passed as-is
        elseif args[i].typ <: CuArray
            # known conversion to CuDeviceArray
        elseif args[i].typ <: CuManaged
            # should be fine -- will be checked in manage_arguments()
        else
            # TODO: warn optionally?
            # TODO: also display variable name, if possible?
            bt = ""
            @static if DEBUG
                bt_raw = backtrace()
                bt = sprint(io->Base.show_backtrace(io, bt_raw))
            end
            warn("you passed an unmanaged $(args[i].typ) argument -- assuming input/output (costly!)$bt\n")
            args[i] = ArgRef(CuInOut{args[i].typ}, :( CuInOut($(args[i].ref)) ))
        end
    end

    return args
end

# Replace CuManaged-wrapped arguments by the underlying argument it manages,
# managing data up- and downloading along the way
function manage_arguments(args::Array{ArgRef})
    setup = Expr[]
    managed_args = Array(ArgRef, length(args))
    teardown = Expr[]

    # TODO: accessing args[i].ref reconstructs the managed container
    #       avoid by stripping Cu*() and accessing the underlying var?

    for i in 1:length(args)
        if args[i].typ <: CuManaged
            input = (args[i].typ <: CuIn) || (args[i].typ <: CuInOut)
            output = (args[i].typ <: CuOut) || (args[i].typ <: CuInOut)

            managed_type = eltype(args[i].typ)
            if managed_type <: Array
                # set-up
                managed_var = gensym("arg$(i)")
                if input
                    # TODO: CuArray shouldn't auto-upload! Use to_device or smth
                    #       maybe upload/download(CuArray)
                    push!(setup, :( $managed_var = CuArray($(args[i].ref).data) ))
                else
                    # create without initializing
                    push!(setup, :( $managed_var =
                        CuArray($(eltype(managed_type)),
                                size($(args[i].ref).data)) ))
                end

                # TODO: N=1 -- how to / does it support higher dimensions?
                #       does this even make sense?
                managed_args[i] = ArgRef(CuArray{eltype(managed_type), 1},
                                         managed_var)

                # tear-down
                if output
                    data_var = gensym("arg$(i)_data")
                    append!(teardown, [
                        :( $data_var = to_host($managed_var) ),
                        :( copy!($(args[i].ref).data, $data_var) ) ])
                end
                push!(teardown, :(free($managed_var)))
            elseif isbits(managed_type)
                error("managed bits types are not supported -- use an array instead")
            else
                # TODO: support more types, for example CuOut(status::Int)
                error("invalid managed type -- cannot handle $managed_type")
            end
        else
            managed_args[i] = args[i]
        end
    end

    return (setup, managed_args, teardown)
end

# Convert the arguments to be used in combination with @cucall
function convert_arguments(args::Array{ArgRef})
    converted_args = Array(ArgRef, length(args))

    for i in 1:length(args)
        if args[i].typ <: DevicePtr
            # we currently don't generate code with device pointers,
            # so get a hold of the inner, raw Ptr value
            converted_args[i] = ArgRef(Ptr{eltype(args[i].typ)},
                                      :( $(args[i].ref).inner ))
        elseif isbits(args[i].typ)
            # pass these as-is

            if args[i].typ <: Ptr
                warn("passing a regular pointer to a device function")
            end

            converted_args[i] = args[i]
        elseif args[i].typ <: CuArray
            # TODO: pass a CuDeviceArray-wrapped pointer,
            #       so our type matches the actual argument?
            # NOTE: when wrapping ptr in a CuDeviceArray,
            #       the type would now be Void because of Ptr{Void}
            #       (no constructor accepting a pointer?)
            # TODO: warn if the eltype is Any?
            converted_args[i] = ArgRef(CuDeviceArray{eltype(args[i].typ)},
                                      :( $(args[i].ref).ptr.inner ))
        else
            error("invalid argument type -- cannot handle $(args[i].typ)")
        end

    end

    return converted_args
end

# FIXME: the entire code_* logic is broken now that they check for being called
#        from within a staged function.
function get_function_module{F<:Function}(ftype::Type{F}, types::Type...)
    debug("Compiling $ftype$(types)")

    # Generate LLVM IR
    # NOTE: this is lifted from reflection.jl::_dump_function
    t = Base.tt_cons(ftype, Base.to_tuple_type(types))
    # Next call will trigger compilation (if necessary)
    llvmf = ccall(:jl_get_llvmf, Ptr{Void}, (Any, Bool, Bool), t, false, true)
    if llvmf == C_NULL
        error("no method found for kernel $ftype with argument types $args_type")
    end

    # Generate (PTX) assembly
    module_asm = ccall(:jl_dump_function_asm, Any, (Ptr{Void},Cint), llvmf, 0)::String

    # Get entry point
    module_entry = ccall(:jl_dump_function_name, Any, (Ptr{Void},), llvmf)::String

    # Try to get a hold of the original function
    # NOTE: this doesn't work for closures, as there can be multiple instances
    #       with each a different environment
    f = Nullable{Function}()
    try
        f = Nullable{Function}(ftype.instance)
    end

    # FIXME: put this before the IR/PTX generation when #14942 is fixed
    @static if TRACE
        if !isnull(f) && false
            trace("Lowered AST:\n$(code_lowered(f, types))")
            trace("Typed AST (::ANY types shown in red):\n")
            code_warntype(STDERR, get(f), types)
        end
    end

    trace("Function entry point: $module_entry")

    # DEBUG: dump the LLVM IR
    # TODO: this doesn't contain the full call cycle
    @static if TRACE
        if !isnull(f) && false
            # Generate a safe and unique name
            function_uid = "$(get(f))-"
            if length(types) > 0
                function_uid *= join([replace(string(x), r"\W", "")
                                    for x in types], '.')
            else
                function_uid *= "Void"
            end

            buf = IOBuffer()
            code_llvm(buf, get(f), types)
            module_llvm = bytestring(buf)

            output = "$(dumpdir[])/$function_uid.ll"
            if isfile(output)
                warn("Could not write LLVM IR to $output (file already exists !?)")
            else
                open(output, "w") do io
                    write(io, module_llvm)
                end
                trace("Wrote kernel LLVM IR to $output")
            end
        end
    end

    # DEBUG: dump the (PTX) assembly
    @static if TRACE && false
        output = "$(dumpdir[])/$function_uid.ptx"
        if isfile(output)
            warn("Could not write (PTX) assembly to $output (file already exists !?)")
        else
            open(output, "w") do io
                write(io, module_asm)
            end
            trace("Wrote kernel (PTX) assembly to $output")
        end
    end

    # Return module & module entry containing function
    return (module_asm, module_entry)
end

# TODO: can we do this in the compiler?
const code_cache = Dict{Tuple{Type, Tuple}, Tuple{AbstractString, AbstractString}}()
const func_cache = Dict{Tuple{Type, Tuple}, CuFunction}()

@generated function get_kernel{F<:Function}(ftype::F, types::Any...)
    key = (ftype, types)

    if (haskey(CUDAnative.code_cache, key))
        (module_asm, module_entry) = CUDAnative.code_cache[key]
    else
        (module_asm, module_entry) = get_function_module(ftype, types...)
        CUDAnative.code_cache[key] = (module_asm, module_entry)
    end

    @gensym ptx_func ptx_mod
    kernel_compilation = quote
        if (haskey(CUDAnative.func_cache, $key))
            $ptx_func = CUDAnative.func_cache[$key]
        else
            $ptx_mod = CuModule($module_asm)
            $ptx_func = CuFunction($ptx_mod, $module_entry)
            CUDAnative.func_cache[$key] = $ptx_func
        end
        return $ptx_func
    end
end

# Construct the necessary argument conversions for launching a PTX kernel
# with given Julia arguments
# TODO: we should split this in a performance oriented and debugging version
@generated function generate_launch{F<:Function}(config::Tuple{CuDim, CuDim, Int},
                                                 ftype::F, argspec::Any...)
    # Process the arguments
    outer_args = read_arguments(argspec)
    (managing_setup, managed_args, managing_teardown) = manage_arguments(outer_args)
    args = convert_arguments(managed_args)

    # TODO: re-use get_kernel here

    types = tuple([arg.typ for arg in args]...)
    key = (ftype, types)

    if haskey(CUDAnative.code_cache, key)
        (module_asm, module_entry) = CUDAnative.code_cache[key]
    else
        (module_asm, module_entry) = get_function_module(ftype, types...)
        CUDAnative.code_cache[key] = (module_asm, module_entry)
    end

    @gensym ptx_func ptx_mod
    kernel_compilation = quote
        if (haskey(CUDAnative.func_cache, $key))
            $ptx_func = CUDAnative.func_cache[$key]
        else
            $ptx_mod = CuModule($module_asm)
            $ptx_func = CuFunction($ptx_mod, $module_entry)
            CUDAnative.func_cache[$key] = $ptx_func
        end
    end

    refs = Expr(:tuple, [arg.ref for arg in args]...)
    kernel_call = :( CUDAnative.exec(config, $ptx_func, $refs) )

    # Throw everything together
    exprs = Expr(:block)
    append!(exprs.args, managing_setup)
    append!(exprs.args, kernel_compilation.args)
    push!(exprs.args, kernel_call)
    append!(exprs.args, managing_teardown)
    return exprs
end

# Perform the run-time API calls launching the kernel
function exec(config::Tuple{CuDim, CuDim, Int}, ptx_func::CuFunction, args::Tuple{Vararg{Any}})
    grid  = config[1]
    block = config[2]
    shared_bytes = config[3]

    # Launch the kernel
    launch(ptx_func, grid, block, args, shmem_bytes=shared_bytes)
end
