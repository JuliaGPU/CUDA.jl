# Native execution support

export
    @cuda


#
# macros/functions for native Julia-CUDA processing
#

immutable TypeConst{val} end
value{val}(::Type{TypeConst{val}}) = val

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
    kernel_func_sym = callexpr.args[1]
    if !isa(kernel_func_sym, Symbol)
        # NOTE: cannot support Module.Kernel calls, because TypeConst only works
        #       on symbols (why?). If it allowed Expr's or even Strings (through
        #       string(expr)), we could specialize the generated function on
        #       that.
        throw(ArgumentError("only simple function calls are supported"))
    end

    # HACK: wrap the function symbol in a type, so we can specialize on it in
    #       the generated function
    kernel_func_const = TypeConst{kernel_func_sym}()
    # TODO: insert some typeasserts? @cuda ([1,1], [1,1]) now crashes
    esc(Expr(:call, CUDA.generate_launch, config, kernel_func_const,
             callexpr.args[2:end]...))
end

# TODO: is this necessary? Just check the method cache?
func_cache = Dict{Tuple{Symbol, Tuple}, CuFunction}()

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
            warn("you passed an unmanaged $(args[i].typ) argument -- assuming input/output (costly!)")
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

# Construct the necessary argument conversions for launching a PTX kernel
# with given Julia arguments
# TODO: we should split this in a performance oriented and debugging version
@generated function generate_launch(config::Tuple{CuDim, CuDim, Int},
                                    func_const::TypeConst, argspec::Any...)
    exprs = Expr(:block)

    # Process the arguments
    #
    # FIXME: for some reason, this generated function runs multiple times, with
    #        different sets of increasingly typed arguments. For some reason,
    #        those partially untyped executions halt somewhere in
    #        manage_arguments, and only the final, fully typed invocation
    #        actually gets to compile...
    #
    # NOTE: the above is why there are so many "unmanaged type" errors
    args = read_arguments(argspec)
    (managing_setup, managed_args, managing_teardown) = manage_arguments(args)
    kernel_args = convert_arguments(managed_args)

    # Compile the function
    kernel_specsig = tuple([arg.typ for arg in kernel_args]...)
    kernel_func_sym = value(func_const)
    if haskey(func_cache, (kernel_func_sym, kernel_specsig))
        trace("Cache hit for $kernel_func_sym$(kernel_specsig)")
        ptx_func = func_cache[kernel_func_sym, kernel_specsig]
    else
        debug("Compiling $kernel_func_sym$(kernel_specsig)")
        kernel_func = eval(:($(current_module()).$kernel_func_sym))

        # TODO: get hold of the IR _before_ calling jl_to_ptx (which does
        # codegen + asm gen)

        # TODO: manual jl_to_llvmf now that it works
        trace("Generating LLVM IR and PTX")
        t = Base.tt_cons(Core.Typeof(kernel_func), Base.to_tuple_type(kernel_specsig))
        (module_ptx, module_entry) = ccall(:jl_to_ptx,
                Any, (Any, Any), kernel_func, t
            )::Tuple{AbstractString, AbstractString}

        # FIXME: put this before the IR/PTX generation when #14942 is fixed
        trace("Lowered AST:\n$(code_lowered(kernel_func, kernel_specsig))")
        trace("Typed AST (::ANY types shown in red):\n")
        if TRACE[]
            code_warntype(STDERR, kernel_func, kernel_specsig)
        end

        trace("Kernel entry point: $module_entry")

        # DEBUG: dump the LLVM IR
        # TODO: this doesn't contain the full call cycle
        if TRACE[]
            # Generate a safe and unique name
            kernel_uid = "$(kernel_func)-"
            if length(kernel_specsig) > 0
                kernel_uid *= join([replace(string(x), r"\W", "")
                                    for x in kernel_specsig], '.')
            else
                kernel_uid *= "Void"
            end

            buf = IOBuffer()
            code_llvm(buf, kernel_func, kernel_specsig)
            module_llvm = bytestring(buf)

            output = "$(dumpdir[])/$kernel_uid.ll"
            if isfile(output)
                warn("Could not write LLVM IR to $output (file already exists !?)")
            else
                open(output, "w") do io
                    write(io, module_llvm)
                end
                trace("Wrote kernel LLVM IR to $output")
            end
        end

        # DEBUG: dump the PTX assembly
        # TODO: make code_native return PTX code
        if TRACE[]
            output = "$(dumpdir[])/$kernel_uid.ptx"
            if isfile(output)
                warn("Could not write PTX assembly to $output (file already exists !?)")
            else
                open(output, "w") do io
                    write(io, module_ptx)
                end
                trace("Wrote kernel PTX assembly to $output")
            end
        end

        # Create CUDA module
        ptx_mod = try
            # TODO: what with other kernel calls? entirely new module? or just
            # add them?
            CuModule(module_ptx)
        catch err
            if err == ERROR_NO_BINARY_FOR_GPU
                # Usually the PTX code is invalid, so try to assembly using
                # "ptxas" manually in order to get some more information
                try
                    readall(`ptxas`)
                    (path, io) = mkstemps(".ptx")
                    print(io, module_ptx)
                    close(io)
                    # FIXME: get a hold of the actual architecture (see cgctx)
                    run(ignorestatus(`ptxas --gpu-name=sm_35 $path`))
                    rm(path)
                end
            end
            rethrow(err)
        end

        # Get CUDA function object
        ptx_func = CuFunction(ptx_mod, module_entry)

        # Cache result to avoid unnecessary compilation
        func_cache[(kernel_func_sym, kernel_specsig)] = ptx_func
    end

    # Throw everything together and generate the final runtime call
    append!(exprs.args, managing_setup)
    kernel_arg_expr = Expr(:tuple, [arg.ref for arg in kernel_args]...)
    push!(exprs.args, :( CUDA.exec(config, $ptx_func, $kernel_arg_expr) ))
    append!(exprs.args, managing_teardown)

    exprs
end

# Perform the run-time API calls launching the kernel
function exec(config::Tuple{CuDim, CuDim, Int}, ptx_func::CuFunction, args::Tuple{Vararg{Any}})
    grid  = config[1]
    block = config[2]
    shared_bytes = config[3]

    # Launch the kernel
    launch(ptx_func, grid, block, args, shmem_bytes=shared_bytes)
end
