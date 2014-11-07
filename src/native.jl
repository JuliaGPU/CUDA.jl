# Native support for cuda


#
# macros/functions for native Julia-CUDA processing
#

func_dict = Dict{(Function, Tuple), CuFunction}()

# User-friendly macro wrapper
# @cuda (dims...) kernel(args...) -> CUDA.exec((dims...), kernel, [args...])
macro cuda(config, callexpr::Expr)
	esc(Expr(:call, CUDA.exec, config, callexpr.args[1], Expr(:cell1d, callexpr.args[2:end]...)))
end

function exec(config, func::Function, args::Array{Any})
	jl_m::Module = config[1]
	grid::CuDim  = config[2]
	block::CuDim = config[3]
	shared_bytes::Int = length(config) > 3 ? config[4] : 0

	# Check argument type (should be either managed or on-device already)
	for it in enumerate(args)
		i = it[1]
		arg = it[2]

		# TODO: create a CuAddressable hierarchy rather than checking for each
		#       type (currently only CuArray) individually?
		#       Maybe based on can_convert_to(CuPtr)?
		if !isa(arg, CuManaged) && !isa(arg, CuPtr)&& !isa(arg, CuArray)
			warn("You specified an unmanaged host argument -- assuming input/output")
			args[i] = CuInOut(arg)
		end
	end

	# Prepare arguments (allocate memory and upload inputs, if necessary)
	args_jl_ty = Array(Type, length(args))	# types to codegen the kernel for
	args_cu = Array(Any, length(args))		# values to pass to that kernel
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
		elseif isa(arg, CuPtr)
			args_jl_ty[i] = typeof(arg)
			args_cu[i] = arg
		else
			error("Cannot handle arguments of type $(typeof(arg))")
		end
	end

	# Cached kernel compilation
	if haskey(func_dict, (func, tuple(args_jl_ty...)))
		cuda_func = func_dict[func, tuple(args_jl_ty...)]
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
		moduleString = code_native_module("cuda")

		# create cuda module
		cu_m = try
			CuModule(module_ptx)
		catch err
			if isa(err, CuDriverError) && err.code == 209
				# CUDA_ERROR_NO_BINARY_FOR_GPU (#209) usually indicates the PTX
				# code was invalid, so try to assembly using "ptxas" manually in
				# order to get some more information
				try
					readall(`ptxas`)
					(path, io) = mktemp()
					print(io, module_ptx)
					close(io)
					# TODO: don't hardcode sm_20
					output = readall(`ptxas --gpu-name=sm_20 $path`)
					warn(output)
					rm(path)
				end
			end
			throw(err)
		end

		# Get internal function name
		internal_name = function_name_llvm(jl_m, func, tuple(args_jl_ty...))
		# Get cuda function object
		cuda_func = CuFunction(cu_m, internal_name)

		# Cache result to avoid unnecessary compilation
		func_dict[(func, tuple(args_jl_ty...))] = cuda_func
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
