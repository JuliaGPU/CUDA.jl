# Native support for cuda


#
# transfer datatypes
#

type CuIn{T}
	data::T
end
length(i::CuIn) = length(i.data)
size(i::CuIn) = size(i.data)
eltype{T}(i::CuIn{T}) = T

type CuOut{T}
	data::T
end
length(o::CuOut) = length(o.data)
size(o::CuOut) = size(o.data)
eltype{T}(o::CuOut{T}) = T

type CuInOut{T}
	data::T
end
length(io::CuInOut) = length(io.data)
size(io::CuInOut) = size(io.data)
eltype{T}(io::CuInOut{T}) = T


#
# macros/functions for native julia-cuda processing
#

func_dict = Dict{(Function, Tuple), CuFunction}()

# User-friendly macro wrapper
# @cuda (dims...) kernel(args...) -> CUDA.exec((dims...), kernel, args...)
macro cuda(config, callexpr::Expr)
	esc(Expr(:call, CUDA.exec, config, callexpr.args...))
end

function exec(config, func::Function, args...)
	jl_m::Module = config[1]
	grid::CuDim  = config[2]
	block::CuDim = config[3]
	shared_bytes::Int = length(config) > 3 ? config[4] : 0

	# Process arguments
	args_jl_ty = Array(Type, length(args))	# types to codegen the kernel for
	args_cu = Array(Any, length(args))		# values to pass to that kernel
	for it in enumerate(args)
		i = it[1]
		arg = it[2]

		# TODO: no CuIn, only Out or InOut meaning the data has to be read back.
		#       then unconditionally copy inputs when arrays
		if isa(arg, CuIn) || isa(arg, CuInOut)
			arg_el = arg.data
			arg_el_type = eltype(arg)
			if arg_el_type <: Array
				args_jl_ty[i] = Ptr{eltype(arg_el_type)}
				args_cu[i] = CuArray(arg_el)
			elseif arg_el_type <: CuArray
				args_jl_ty[i] = Array{eltype(arg_el), ndims(arg_el)}
				args_cu[i] = arg_el
			else
				error("No support for $arg_el_type input values")
			end
		elseif isa(arg, CuOut)
			arg_el = arg.data
			arg_el_type = eltype(arg)
			if arg_el_type <: Array
				# println("Array")
				args_jl_ty[i] = Ptr{eltype(arg_el_type)}
				args_cu[i] = CuArray(eltype(arg_el), size(arg_el))
			elseif arg_el_type <: CuArray
				# println("CuArray")
				args_jl_ty[i] = Array{eltype(arg_el), ndims(arg_el)}
				args_cu[i] = arg_el
			else
				error("No support for $arg_el_type output values")
			end
		else
			# Other type
			# TODO: can we check here already if the type will require boxing?
			warn("No explicit support for $(typeof(arg)) input values; passing as-is")
			args_jl_ty[i] = typeof(arg)
			args_cu[i] = arg
		end
	end

	# conditional compilation of function
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

	# Launch cuda object
	launch(cuda_func, grid, block, tuple(args_cu...), shmem_bytes=shared_bytes)

	# Get results
	for it in enumerate(args)
		i = it[1]
		arg = it[2]
		if isa(arg, CuOut) || isa(arg, CuInOut)
			if isa(arg.data, Array)
				host = to_host(args_cu[i])
				copy!(arg.data, host)
			elseif isa(arg.data, CuArray)
				#println("Copy to CuArray")
			end
		end
	end

	# Free memory
	# TODO: merge with previous
	for it in enumerate(args)
		i = it[1]
		arg = it[2]
		if eltype(arg) <: Array
			free(args_cu[i])
		end
	end
end
