# Native execution support

export
    @cuda,
    initialize_codegen


# NOTE: keep this in sync with the architectures supported by NVPTX
#       (see lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc)
const architectures = [
    (v"2.0", "sm_20"),
    (v"2.1", "sm_21"),
    (v"3.0", "sm_30"),
    (v"3.5", "sm_35"),
    (v"5.0", "sm_50") ]

# TODO: allow code generation without actual device/ctx?
type CuCodegenContext
    ctx::CuContext
    dev::CuDevice
    triple::String
    arch::String

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
            cap = capability(dev)
            if cap < architectures[1][1]
                error("No support for SM < $(architectures[1][1])")
            end
            for i = 2:length(architectures)
                if cap < architectures[i][1]
                    arch = architectures[i-1][2]
                    break
                end
            end
        end

        ccall(:jl_init_ptx_codegen, Void, (String, String), triple, arch)

        new(ctx, dev, triple, arch)
    end
end
cgctx = Nullable{CuCodegenContext}()
function initialize_codegen(ctx::CuContext, dev::CuDevice)
    # TODO: this is ugly. If we allow multiple contexts, codegen.cpp will
    #       have a map of Ctx=>PM/FPM/M of some sort. This is too PTX-
    #       specific, and should tie into @target somehow
    global cgctx
    if !Base.isnull(cgctx)
        error("Cannot have multiple active code generation contexts")
    end
    cgctx = Nullable{CuCodegenContext}(CuCodegenContext(ctx, dev))
end


#
# macros/functions for native Julia-CUDA processing
#

immutable FunctionConst{sym} end
symbol{sym}(::Type{FunctionConst{sym}}) = sym

# The @cuda macro provides a user-friendly wrapper for prepare_exec(). It is
# executed during parsing, and substitutes calls to itself with properly
# formatted calls to prepare_exec().
macro cuda(config::Expr, callexpr::Expr)
    # Sanity checks
    if config.head != :tuple || !(2 <= length(config.args) <= 3)
        error("first argument to @cuda should be a tuple (gridDim, blockDim, [shmem])")
    end
    if length(config.args) == 2
        push!(config.args, :0)
    end
    if callexpr.head != :call
        error("second argument to @cuda should be a fully specified function call")
    end
    func_sym = callexpr.args[1]
    if func_sym.head != :.
        error("kernel function call should be fully specified, including the module")
    end
    kernel_mod_sym = func_sym.args[1]
    kernel_func_sym = func_sym.args[2]
    # FIXME: what is this :quote?
    @assert kernel_func_sym.head == :quote && length(kernel_func_sym.args) == 1
    kernel_func_sym = kernel_func_sym.args[1]

    # Get a hold of the module and function
    calling_mod = current_module()
    kernel_mod = try
        eval(:( $calling_mod.$kernel_mod_sym ))
    catch
        error("could not inspect module $kernel_mod_sym -- have you imported it?")
    end
    if !(kernel_func_sym in names(kernel_mod))
        error("could not find function $kernel_func_sym in module $kernel_mod_sym -- is the function exported?")
    end

    # HACK: wrap the function symbol in a type, so we can specialize on it in
    #       the staged function
    kernel_func_const = FunctionConst{kernel_func_sym}()
    esc(Expr(:call, CUDA.prepare_exec, config, kernel_func_const,
             callexpr.args[2:end]...))
end

# TODO: is this necessary? Just check the method cache?
func_cache = Dict{(Symbol, Tuple), CuFunction}()

# The prepare_exec() staged function prepares the call to exec(), and runs once
# for each combination of a call site and type inferred arguments. Knowing the
# input type of each argument, it generates expressions which prepare the inputs
# to be used in combination with a GPU kernel.
stagedfunction prepare_exec(config::(CuDim, CuDim, Int), # NOTE: strangely works
                            func_const::FunctionConst, host_args...)
    exprs = Expr(:block)

    # Sanity checks
    if Base.isnull(cgctx)
        error("native code generation is not initialized yet")
    end

    # Check argument types (should be either managed or on-device already)
    host_args_sym  = Array(Union(Symbol,Expr),      # symbolic reference to the
                           length(host_args))       #  host variable or value
    host_args_type = Array(Type, length(host_args)) # its original type
    for i in 1:length(host_args)
        var = :( host_args[$i] )        # symbolic reference to the argument
        vartype = host_args[i]::Type    # its type

        if !(vartype <: CuManaged) && !(vartype <: DevicePtr{Void}) && !(vartype <: CuArray)
            warn("You specified an unmanaged host argument -- assuming input/output (costly!)")
            newvar = gensym()
            push!(exprs.args, :( $newvar = CuInOut($var) ))
            var = newvar
            vartype = CuInOut{vartype}
        end

        host_args_sym[i] = var
        host_args_type[i] = vartype
    end

    # Prepare arguments (allocate memory and upload inputs, if necessary)
    kernel_args_sym  = Array(Union(Symbol,Expr),     # input objects for launch()
                             length(host_args))      #  (var symbol or value expr)
    kernel_args_type = Array(Type,                   # types for precompile()
                             length(host_args))      #  and launch()
    for i in 1:length(host_args)
        host_arg_sym = host_args_sym[i]
        host_arg_type = host_args_type[i]

        # Process the argument based on its type
        if host_arg_type <: CuManaged
            input = (host_arg_type <: CuIn) || (host_arg_type <: CuInOut)

            if eltype(host_arg_type) <: Array
                host_arg_type = Ptr{eltype(eltype(host_arg_type))}
                if input
                    newvar = gensym()
                    push!(exprs.args, :( $newvar = CuArray($host_arg_sym.data) ))
                    host_arg_sym = newvar
                else
                    # create without initializing
                    newvar = gensym()
                    push!(exprs.args, :( $newvar = CuArray(eltype($host_arg_sym.data),
                                                           size($host_arg_sym.data)) ))
                    host_arg_sym = newvar
                end
            else
                warn("no explicit support for $(typeof(arg)) input values; passing as-is")
                host_arg_type = eltype(host_arg_type)
                host_arg_sym = :( $host_arg_sym.data )
            end
        elseif host_arg_type <: CuArray
            host_arg_type = Ptr{eltype(host_arg_type)}
            # launch() converts CuArrays to DevicePtrs using ptrbox
            # FIXME: replace with inner pointer ourselves?
        elseif host_arg_type <: DevicePtr{Void}
            # we can use these as-is
        else
            error("cannot handle arguments of type $(host_arg_type)")
        end

        kernel_args_type[i] = host_arg_type
        kernel_args_sym[i] = :( $host_arg_sym )
    end

    # Compile the function
    kernel_specsig = tuple(kernel_args_type...)
    kernel_func_sym = symbol(func_const)
    kernel_func = eval(:($(current_module()).$kernel_func_sym))
    if haskey(func_cache, (kernel_func_sym, kernel_specsig))
        ptx_func = func_cache[kernel_func_sym, kernel_specsig]
    else
        # trigger function compilation
        try
            precompile(kernel_func, kernel_specsig)
        catch err
            print("\n\n\n*** Compilation failed ***\n\n")
            # this is most likely caused by some boxing issue, so dump the ASTs
            # to help identifying the offending variable
            print("-- lowered AST --\n\n", code_lowered(kernel_func, kernel_specsig), "\n\n")
            print("-- typed AST --\n\n", code_typed(kernel_func, kernel_specsig), "\n\n")
            throw(err)
        end

        # trigger module compilation
        module_ptx = ccall(:jl_to_ptx, Any, ())::String

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
                    run(`ptxas --gpu-name=$(get(cgctx).arch) $path`)
                    rm(path)
                end
            end
            throw(err)
        end

        # Get internal function name
        # FIXME: just get new module / clean slate for every CUDA
        # module, with a predestined function name for the kernel? But
        # then what about type specialization?
        ptx_func_name = ccall(:jl_dump_function_name, Any, (Any, Any),
                              kernel_func, kernel_specsig)

        # Get CUDA function object
        ptx_func = CuFunction(ptx_mod, ptx_func_name)

        # Cache result to avoid unnecessary compilation
        func_cache[(kernel_func_sym, kernel_specsig)] = ptx_func
    end

    # Call the runtime part of the execution
    kernel_args = Expr(:tuple, kernel_args_sym...)
    push!(exprs.args, :( CUDA.exec(config, $ptx_func, $kernel_args) ))

    # Finish up (fetch results and free memory, if necessary)
    for i in 1:length(host_args)
        host_arg_sym = host_args_sym[i]
        host_arg_type = host_args_type[i]
        kernel_arg_sym = kernel_args_sym[i]

        if host_arg_type <: CuManaged
            output = (host_arg_type <: CuOut) || (host_arg_type <: CuInOut)

            if eltype(host_arg_type) <: Array
                if output
                    data = gensym()
                    push!(exprs.args, quote
                        $data = to_host($kernel_arg_sym)
                        copy!($host_arg_sym.data, $data)
                    end)
                end
                push!(exprs.args, :(free($kernel_arg_sym)))
            end
        end
    end

    exprs
end

# The exec() function is executed for each kernel invocation, and performs the
# necessary driver interactions to upload and launch the kernel.
function exec(config::(CuDim, CuDim, Int), ptx_func::CuFunction, args::Tuple)
    grid  = config[1]
    block = config[2]
    shared_bytes = config[3]

    # Launch the kernel
    launch(ptx_func, grid, block, args, shmem_bytes=shared_bytes)
end
