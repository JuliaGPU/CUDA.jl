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

macro cuda(config, call::Expr)
	exprs = ()

	# Generate expressions to process the arguments
	@gensym args
	# TODO: fixed-size array
	exprs = :($exprs; $args = Array(Any, 0))
	for arg = call.args[2:end]
		exprs_arg = quote
			push!($args, $(esc(arg)))
		end
		exprs = :($exprs; $exprs_arg)
	end

	# Now execute the function and return all the expressions
	:($exprs; $:(__cuda_exec($(esc(config)), $(esc(call.args[1])), $args...)))
end

function __cuda_exec(config, func::Function, args...)
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
				push!(args_jl_ty, arg_el_type)
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
				push!(args_jl_ty, arg_el_type)
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
		precompile(func, tuple(args_jl_ty...))

		# trigger module compilation
		moduleString = code_native_module("cuda")

		# create cuda module
		cu_m = CuModule(moduleString)

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


# Not used currently
macro __cuda(config, call::Expr)
	# Module and thread/block config
	@gensym md
	@gensym grid
	@gensym block
	exprs = quote
		$md    = $(esc(config.args[1]))
		$grid  = $(esc(config.args[2]))
		$block = $(esc(config.args[3]))
	end

	# Function
	@gensym func
	expr_func = quote
		$func = $(esc(call.args[1]))
	end
	exprs = :($exprs; $expr_func)

	# Arguments
	args = call.args[2:end]
	@gensym args_jl_ty
	@gensym args_cu
	expr_init = quote
		$args_jl_ty = []
		$args_cu = []
	end
	exprs = :($exprs; $expr_init)

	# Generate expressions to process the arguments
	for arg = args
		@gensym embedded_arg
		exprs_arg = quote
			# if isa($(esc(arg)), CuIn)
			# 	println("Input argument")
			# elseif isa($(esc(arg)), CuOut)
			# 	println("Output argument")
			# elseif isa($(esc(arg)), CuInOut)
			# 	println("In- and output argument")
			# else
			# 	error("Datatype not supported: ", typeof($(esc(arg))))
			# end
			$embedded_arg = $(esc(arg)).data
			if isa($(embedded_arg), CuArray)
				$args_cu = [$args_cu, $(embedded_arg)]
				$args_jl_ty = [$args_jl_ty, Array{eltype($(embedded_arg)), ndims($(embedded_arg))}]
			elseif isa($(embedded_arg), Array)
				$args_jl_ty = [$args_jl_ty, typeof($(embedded_arg))]
				$args_cu = [$args_cu, CuArray($(embedded_arg))]
			end
		end
		exprs = :($exprs; $exprs_arg)
	end

	# Generate expression for retrieving native function name
	@gensym native_name
	@gensym cu_func
	expr_name = quote
		$native_name = function_name_llvm($func, tuple($args_jl_ty...))
		$cu_func = CuFunction($md, $native_name)
		launch($cu_func, $grid, $block, tuple($args_cu...))
	end
	exprs = :($exprs; $expr_name)

	idx = 0
	for arg in args
		idx = idx + 1
		if arg.args[1] == :CuOut || arg.args[1] == :CuInOut
			expr_to_host = quote
				$(esc(arg.args[2])) = to_host($args_cu[$idx])
			end
			exprs = :($exprs; $expr_to_host)
		end
	end

	:($exprs)
end
