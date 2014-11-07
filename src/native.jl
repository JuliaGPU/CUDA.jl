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

# Module definition and usage helper
#
# This just prepends __ptx__ to the module name, a magic string we use to
# identify the module from within the compiler
macro cumodule(e::Expr)
    if e.head == :module
        name_index = 2
    elseif e.head == :using
        name_index = 1
    else
        error("@cumodule can only be used before 'module' or 'using' keywords")
    end

    modname = e.args[name_index]
    e.args[name_index] = symbol("__ptx__" * string(modname))

    esc(Expr(:toplevel, e))
end

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
	args_jl_ty = Array(Type, 0)
	args_cu = Array(Any, 0)
	for arg in args
		if isa(arg, CuIn) || isa(arg, CuInOut)
			arg_el = arg.data
			arg_el_type = eltype(arg)
			if arg_el_type <: Array
				# println("Array")
				push!(args_jl_ty, Ptr{eltype(arg_el_type)})
				push!(args_cu, CuArray(arg_el))
			elseif arg_el_type <: CuArray
				# println("CuArray")
				push!(args_jl_ty, Array{eltype(arg_el),
				      ndims(arg_el)})
				push!(args_cu, arg_el)
			else
				# Other element type
			end
		elseif isa(arg, CuOut)
			arg_el = arg.data
			arg_el_type = eltype(arg)
			if arg_el_type <: Array
				# println("Array")
				push!(args_jl_ty, Ptr{eltype(arg_el_type)})
				push!(args_cu, CuArray(eltype(arg_el),
				      size(arg_el)))
			elseif arg_el_type <: CuArray
				# println("CuArray")
				push!(args_jl_ty, Array{eltype(arg_el),
				      ndims(arg_el)})
				push!(args_cu, arg_el)
			else
				# Other element type
			end
		else
			# Other type
			# should not be allowed?
			push!(args_jl_ty, typeof(arg))
			push!(args_cu, arg)
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
	index = 1
	for arg in args
		if isa(arg, CuOut) || isa(arg, CuInOut)
			if isa(arg.data, Array)
				host = to_host(args_cu[index])
				copy!(arg.data, host)
			elseif isa(arg.data, CuArray)
				#println("Copy to CuArray")
			end
		end
		index = index + 1
	end

	# Free memory
	index = 1
	for arg in args
		if eltype(arg) <: Array
			free(args_cu[index])
		end
		index = index + 1
	end
end
