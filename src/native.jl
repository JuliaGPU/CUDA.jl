# Native support for cuda


#
# intrinsics
#
threadId_x() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
                				ret i32 %1""", Int32, ()) + 1
threadId_y() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() readnone nounwind
                				ret i32 %1""", Int32, ()) + 1
threadId_z() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z() readnone nounwind
                				ret i32 %1""", Int32, ()) + 1


#
# transfer datatypes
#
type In{T}
	data::T
end
length(i::In) = length(i.data)
size(i::In) = size(i.data)
eltype{T}(i::In{T}) = T

type Out{T}
	data::T
end
length(o::Out) = length(o.data)
size(o::Out) = size(o.data)
eltype{T}(o::Out{T}) = T

type InOut{T}
	data::T
end
length(io::InOut) = length(io.data)
size(io::InOut) = size(io.data)
eltype{T}(io::InOut{T}) = T


#
# macros for native julia - cuda processing
#
macro __cuda(config, call::Expr, arg_types::Tuple)
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
