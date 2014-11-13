# Native execution support

export
    @cuda,
    CuCodegenContext

# TODO: allow code generation without actual device/ctx?
codegen_initialized = false
type CuCodegenContext
    ctx::CuContext
    dev::CuDevice

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
            arch = "sm_" * string(cap.major) * string(cap.minor)
        end

        # TODO: this is ugly. If we allow multiple contexts, codegen.cpp will
        #       have a map of Ctx=>PM/FPM/M of some sort. This is too PTX-
        #       specific, and should tie into @target somehow
        global codegen_initialized
        if codegen_initialized
            error("Cannot have multiple active code generation contexts")
        end
        codegen_initialized = true
        ccall(:jl_init_ptx_codegen, Void, (String, String), triple, arch)
    end
end

#
# macros/functions for native Julia-CUDA processing
#

func_cache = Dict{(Function, Tuple), CuFunction}()

# User-friendly macro wrapper
# @cuda (dims...) kernel(args...) -> CUDA.exec((dims...), kernel, [args...])
macro cuda(config, callexpr::Expr)
    esc(Expr(:call, CUDA.exec, config, callexpr.args[1],
             Expr(:cell1d, callexpr.args[2:end]...)))
end

function exec(config, func::Function, args::Array{Any})
    jl_m::Module = config[1]
    grid::CuDim  = config[2]
    block::CuDim = config[3]
    shared_bytes::Int = length(config) > 3 ? config[4] : 0
    global codegen_initialized
    assert(codegen_initialized)

    # Check argument type (should be either managed or on-device already)
    for it in enumerate(args)
        i = it[1]
        arg = it[2]

        # TODO: create a CuAddressable hierarchy rather than checking for each
        #       type (currently only CuArray) individually?
        #       Maybe based on can_convert_to(DevicePtr{Void})?
        if !isa(arg, CuManaged) && !isa(arg, DevicePtr{Void})&& !isa(arg, CuArray)
            warn("You specified an unmanaged host argument -- assuming input/output")
            args[i] = CuInOut(arg)
        end
    end

    # Prepare arguments (allocate memory and upload inputs, if necessary)
    args_jl_ty = Array(Type, length(args))    # types to codegen the kernel for
    args_cu = Array(Any, length(args))        # values to pass to that kernel
    for it in enumerate(args)
        i = it[1]
        arg = it[2]

        if isa(arg, CuManaged)
            input = isa(arg, CuIn) || isa(arg, CuInOut)

            if isa(arg.data, Array)
                args_jl_ty[i] = Ptr{eltype(arg.data)}
                if input
                    args_cu[i] = CuArray(arg.data)
                else
                    # create without initializing
                    args_cu[i] = CuArray(eltype(arg.data), size(arg.data))
                end
            else
                warn("No explicit support for $(typeof(arg)) input values; passing as-is")
                args_jl_ty[i] = typeof(arg.data)
                args_cu[i] = arg.data
            end
        elseif isa(arg, CuArray)
            args_jl_ty[i] = Ptr{eltype(arg)}
            args_cu[i] = arg
        elseif isa(arg, DevicePtr{Void})
            args_jl_ty[i] = typeof(arg)
            args_cu[i] = arg
        else
            error("Cannot handle arguments of type $(typeof(arg))")
        end
    end

    # Cached kernel compilation
    if haskey(func_cache, (func, tuple(args_jl_ty...)))
        cuda_func = func_cache[func, tuple(args_jl_ty...)]
    else
        # trigger function compilation
        try
            precompile(func, tuple(args_jl_ty...))
        catch err
            print("\n\n\n*** Compilation failed ***\n\n")
            # this is most likely caused by some boxing issue, so dump the ASTs
            # to help identifying the boxed variable
            print("-- lowered AST --\n\n", code_lowered(func, tuple(args_jl_ty...)), "\n\n")
            print("-- typed AST --\n\n", code_typed(func, tuple(args_jl_ty...)), "\n\n")
            throw(err)
        end

        # trigger module compilation
        module_ptx = ccall(:jl_to_ptx, Any, ())::String

        # create CUDA module
        cu_m = try
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
                    # TODO: don't hardcode sm_20
                    run(`ptxas --gpu-name=sm_20 $path`)
                    rm(path)
                end
            end
            throw(err)
        end

        # Get internal function name
        # FIXME: just get new module / clean slate for every CUDA
        # module, with a predestined function name for the kernel? But
        # then what about type specialization?
        function_name = ccall(:jl_dump_function_name, Any, (Any, Any),
                              func, tuple(args_jl_ty...))

        # Get CUDA function object
        cuda_func = CuFunction(cu_m, function_name)

        # Cache result to avoid unnecessary compilation
        func_cache[(func, tuple(args_jl_ty...))] = cuda_func
    end

    # Launch the kernel
    launch(cuda_func, grid, block, tuple(args_cu...), shmem_bytes=shared_bytes)

    # Finish up (fetch results and free memory, if necessary)
    for it in enumerate(args)
        i = it[1]
        arg = it[2]

        if isa(arg, CuManaged)
            output = isa(arg, CuOut) || isa(arg, CuInOut)

            if isa(arg.data, Array)
                if output
                    host = to_host(args_cu[i])
                    copy!(arg.data, host)
                end

                free(args_cu[i])
            end
        end
    end
end
