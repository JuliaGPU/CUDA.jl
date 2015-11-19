# Native execution support

export
    @cuda,
    CuCodegenContext

# TODO: allow code generation without actual device/ctx?
# TODO: require passing the codegen context to @cuda, so we can have multiple
#       contexts active, generating for and executing on multiple GPUs
type CuCodegenContext
    ctx::CuContext
    dev::CuDevice
    triple::ASCIIString
    arch::ASCIIString

    CuCodegenContext(ctx::CuContext) = CuCodegenContext(ctx, device(ctx))

    function CuCodegenContext(ctx::CuContext, dev::CuDevice)
        # Determine the triple
        if haskey(ENV, "CUDA_FORCE_GPU_TRIPLE")
            triple = ENV["CUDA_FORCE_GPU_TRIPLE"]
        else
            # TODO: detect 64/32
            triple = "nvptx64-nvidia-cuda"
        end

        # Determine the architecture
        if haskey(ENV, "CUDA_FORCE_GPU_ARCH")
            arch = ENV["CUDA_FORCE_GPU_ARCH"]
        else
            arch = architecture(dev)
        end

        # NOTE: forcibly box to AbstractString because jl_init_codegen_ptx
        #       accepts jl_value_t arguments
        ccall(:jl_init_codegen_ptx, Void, (AbstractString, AbstractString), triple, arch)

        new(ctx, dev, triple, arch)
    end
end

function destroy(ctx::CuCodegenContext)
    # TODO: we don't look at ctx, but we should, in order to prevent the user
    #       from keeping a ctx after destruction, or destroying one context
    #       while expecting to destroy another (cfr. other destroy methods,
    #       asserting obj==active_obj)
    ccall(:jl_destroy_codegen_ptx, Void, ())
end


#
# macros/functions for native Julia-CUDA processing
#

immutable TypeConst{val} end
value{val}(::Type{TypeConst{val}}) = val

macro cuda(config::Expr, callexpr::Expr)
    # Sanity checks
    if config.head != :tuple || !(2 <= length(config.args) <= 3)
        error("first argument to @cuda should be a tuple (gridDim, blockDim, [shmem])")
    end
    if length(config.args) == 2
        push!(config.args, :0)
    end
    if callexpr.head != :call
        error("second argument to @cuda should be a function call")
    end
    kernel_func_sym = callexpr.args[1]
    if !isa(kernel_func_sym, Symbol)
        # NOTE: cannot support Module.Kernel calls, because TypeConst only works
        #       on symbols (why?). If it allowed Expr's or even Strings (through
        #       string(expr)), we could specialize the generated function on
        #       that.
        error("only simple function calls are supported")
    end
    if search(string(kernel_func_sym), "kernel_").start != 1
        error("kernel function should start with \"kernel_\"")
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
            @warn("you passed an unmanaged $(args[i].typ) argument -- assuming input/output (costly!)")
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
                @warn("passing a regular pointer to a device function")
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
                                      :( $(args[i].ref).ptr ))
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
        ptx_func = func_cache[kernel_func_sym, kernel_specsig]
    else
        # trigger function compilation
        kernel_func = eval(:($(current_module()).$kernel_func_sym))
        kernel_err = nothing
        @debug("Invoking Julia compiler for $kernel_func$(kernel_specsig)")
        try
            precompile(kernel_func, kernel_specsig)
        catch err
            kernel_err = err
        end

        # dump the ASTs
        # TODO: dump called functions too?
        @debug("Lowered AST:\n$(code_lowered(kernel_func, kernel_specsig))")
        buf = IOBuffer()
        code_warntype(buf, kernel_func, kernel_specsig)
        @debug("Typed AST:\n$(takebuf_string(buf))")

        if kernel_err != nothing
            @error("Kernel compilation phase 1 failed ($(sprint(showerror, kernel_err)))")

            # FIXME: should the exception be catchable?
            #throw(err)
            quit()
        end

        # check if the function actually exists, by mimicking code_llvm()
        # (precompile silently ignores invalid func/specsig combinations)
        # TODO: is there a more clean way to check this?
        # TODO: use dump_module=true instead of our functionality in jl_to_ptx?
        kernel_llvm = Base._dump_function(kernel_func, kernel_specsig,
                                          false,    #native
                                          false,    # wrapper
                                          false,    # strip metadata
                                          false)    # dump module
        if kernel_llvm == ""
            error("no method found for $kernel_func$kernel_specsig")
        end

        # Get internal function name
        # FIXME: just get new module / clean slate for every CUDA
        # module, with a predestined function name for the kernel? But
        # then what about type specialization?
        kernel_fname = ccall(:jl_dump_function_name, Any, (Any, Any),
                             kernel_func, Tuple{kernel_specsig...})
        kernel_llvm = replace(kernel_llvm, kernel_fname, string(kernel_func))

        # DEBUG: dump the LLVM IR
        if Logging._root.level <= DEBUG
            # Generate a safe and unique name
            kernel_uid = "$(kernel_func)-"
            if length(kernel_specsig) > 0
                kernel_uid *= join([replace(string(x), r"\W", "") for x in kernel_specsig], '.')
            else
                kernel_uid *= "Void"
            end

            scriptdir = dirname(Base.source_path())
            builddir = "$scriptdir/.build"
            if !isdir(builddir)
                mkdir(builddir)
            end

            output = "$builddir/$kernel_uid.ll"
            reference = nothing
            if isfile(output)
                reference = output
                (output, _) = mkstemps(".ll")
            end

            f = open(output, "w")
            write(f, kernel_llvm)
            close(f)

            if reference != nothing
                @debug("Differences in kernel LLVM IR (if any):")
                # TODO: capture output
                run(ignorestatus(`diff -u $reference $output`))
                rm(output)
            else
                @debug("Kernel LLVM IR: $(kernel_llvm)")
            end
        end

        # trigger module compilation
        module_ptx = ccall(:jl_to_ptx, Any, ())::AbstractString

        # DEBUG: dump the PTX assembly
        if Logging._root.level <= DEBUG
            # Extract the kernel function's PTX
            # TODO: do this in LLVM
            kernel_start = searchindex(module_ptx, ".visible .entry $(kernel_fname)")
            kernel_end = search(module_ptx, r"\n}", kernel_start)
            kernel_ptx = module_ptx[kernel_start:kernel_end.start+1]

            kernel_ptx = replace(kernel_ptx, kernel_fname, string(kernel_func))

            output = "$builddir/$kernel_uid.ptx"
            reference = nothing
            if isfile(output)
                reference = output
                (output, _) = mkstemps(".ptx")
            end

            f = open(output, "w")
            write(f, kernel_ptx)
            close(f)

            if reference != nothing
                @debug("Differences in kernel PTX assembly (if any):")
                # TODO: capture output
                x = run(ignorestatus(`diff -u $reference $output`))
                rm(output)
            else
                @debug("Kernel PTX assembly:\n$(kernel_ptx)")
            end
        end

        # create CUDA module
        ptx_mod = try
            # TODO: what with other kernel calls? entirely new module? or just
            # add them?
            CuModule(module_ptx)
        catch err
            if isa(err, CuError) && err.code == 209
                # CUDA_ERROR_NO_BINARY_FOR_GPU (#209) usually indicates the PTX
                # code was invalid, so try to assembly using "ptxas" manually in
                # order to get some more information
                try
                    readall(`ptxas`)
                    (path, io) = mktemp()
                    print(io, module_ptx)
                    close(io)
                    # FIXME: get a hold of the actual architecture (see cgctx)
                    run(ignorestatus(`ptxas --gpu-name=sm_35 $path`))
                    rm(path)
                end
            end
            throw(err)
        end

        # Get CUDA function object
        ptx_func = CuFunction(ptx_mod, kernel_fname)

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
