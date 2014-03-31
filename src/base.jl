# Load & initialize CUDA driver

const libcuda = dlopen("libcuda")


macro cucall(f, argtypes, args...)
	quote
		g = haskey(api_dict, $f) ? api_dict[$f] : $f
		_curet = ccall(dlsym(libcuda, g), Cint, $argtypes, $(args...))
		if _curet != 0
			err = CuDriverError(int(_curet))
			message = description(err)
			throw("CUDA driver error ($(err.code)) : $message")
		end
	end
end

function initialize(api_version::Int)
	populate_api_dict(api_version)
	@cucall(:cuInit, (Cint,), 0)
	println("CUDA Driver Initialized")
end


# Get driver version

function driver_version()
	a = Cint[0]
	@cucall(:cuDriverGetVersion, (Ptr{Cint},), a)
	return int(a[1])
end


# box a variable into array

cubox{T}(x::T) = T[x]


# create dict for ambiguous api calls

global api_dict = (Symbol => Symbol)[]
function populate_api_dict(api_version::Int)
	if api_version >= 3020
		api_dict[:cuDeviceTotalMem] = :cuDeviceTotalMem_v2
		api_dict[:cuCtxCreate]      = :cuCtxCreate_v2
		api_dict[:cuMemAlloc]       = :cuMemAlloc_v2
		api_dict[:cuMemcpyHtoD]     = :cuMemcpyHtoD_v2
		api_dict[:cuMemcpyDtoH]     = :cuMemcpyDtoH_v2
		api_dict[:cuMemFree]        = :cuMemFree_v2
	end
	if api_version >= 4000
		api_dict[:cuCtxDestroy]     = :cuCtxDestroy_v2
		api_dict[:cuCtxPushCurrent] = :cuCtxPushCurrent_v2
		api_dict[:cuCtxPopCurrent]  = :cuCtxPopCurrent_v2
	end
end


initialize(5050)


# macro for native julia - cuda processing

macro cuda(config, call::Expr)
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
			# if isa($(esc(arg)), In)
			# 	println("Input argument")
			# elseif isa($(esc(arg)), Out)
			# 	println("Output argument")
			# elseif isa($(esc(arg)), InOut)
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
		if arg.args[1] == :Out || arg.args[1] == :InOut
			expr_to_host = quote
				$(esc(arg.args[2])) = to_host($args_cu[$idx])
			end
			exprs = :($exprs; $expr_to_host)
		end
	end

	:($exprs)
end
