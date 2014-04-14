# Native support for cuda


#
# intrinsics
#
# Thread ID
threadId_x() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
									   ret i32 %1""", Int32, ()) + 1 # ::Int32 # This gives error
threadId_y() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() readnone nounwind
									   ret i32 %1""", Int32, ()) + 1 # ::Int32 # This gives error
threadId_z() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z() readnone nounwind
									   ret i32 %1""", Int32, ()) + 1
# Block Dim (num threads per block)
numThreads_x() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() readnone nounwind
										ret i32 %1""", Int32, ())
numThreads_y() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() readnone nounwind
										ret i32 %1""", Int32, ())
numThreads_z() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() readnone nounwind
										ret i32 %1""", Int32, ())
# Block ID
blockId_x() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() readnone nounwind
									  ret i32 %1""", Int32, ()) + 1
blockId_y() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() readnone nounwind
									  ret i32 %1""", Int32, ()) + 1
blockId_z() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() readnone nounwind
									  ret i32 %1""", Int32, ()) + 1
# Grid Dim (num blocks per grid)
numBlocks_x() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() readnone nounwind
										ret i32 %1""", Int32, ())
numBlocks_y() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() readnone nounwind
										ret i32 %1""", Int32, ())
numBlocks_z() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() readnone nounwind
										ret i32 %1""", Int32, ())
# Warpsize
warpsize() = Base.llvmcall(false, """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize() readnone nounwind
									 ret i32 %1""", Int32, ())
# Barrier
sync_threads() = Base.llvmcall(false, """call void @llvm.nvvm.barrier0()
										 ret void""", Void, ())


#
# transfer datatypes
#
type CuIn{T<:Array}
	data::T
end
length(i::CuIn) = length(i.data)
size(i::CuIn) = size(i.data)
eltype{T}(i::CuIn{T}) = T

type CuOut{T<:Array}
	data::T
end
length(o::CuOut) = length(o.data)
size(o::CuOut) = size(o.data)
eltype{T}(o::CuOut{T}) = T

type CuInOut{T<:Array}
	data::T
end
length(io::CuInOut) = length(io.data)
size(io::CuInOut) = size(io.data)
eltype{T}(io::CuInOut{T}) = T


#
# shared memory
#
cuSharedMem() = Base.llvmcall(true, """@i = external addrspace(3) global [0 x i32]""", Ptr{Int32}, ())
setCuSharedMem(shmem, index, value) = Base.llvmcall(false,
	"""%4 = tail call i32 addrspace(3)* @llvm.nvvm.ptr.gen.to.shared.p3i32.p0i32( i32* %0 )
	   %5 = getelementptr inbounds i32 addrspace(3)* %4, i64 %1
	   %6 = trunc i64 %2 to i32
	   store i32 %6, i32 addrspace(3)* %5
	   ret void""",
	Void, (Ptr{Int32}, Int64, Int64), shmem, index-1, value)
getCuSharedMem(shmem, index) = Base.llvmcall(false,
	"""%3 = tail call i32 addrspace(3)* @llvm.nvvm.ptr.gen.to.shared.p3i32.p0i32( i32* %0 )
	   %4 = getelementptr inbounds i32 addrspace(3)* %3, i64 %1
	   %5 = load i32 addrspace(3)* %4
	   ret i32 %5""",
	Int32, (Ptr{Int32}, Int,), shmem, index-1)


#
# macros/functions for native julia-cuda processing
#
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
	md::CuModule = config[1]
	grid::CuDim  = config[2]
	block::CuDim = config[3]
	shared_bytes::Int = length(config) > 3 ? config[4] : 0

	# Process arguments
	args_jl_ty = Array(Type, 0)
	args_cu = Array(CuArray, 0)
	for arg in args
		if isa(arg, CuIn) || isa(arg, CuOut) || isa(arg, CuInOut)
			arg_el = arg.data
			arg_el_type = eltype(arg)
			if arg_el_type <: Array
				# println("Array")
				push!(args_jl_ty, arg_el_type)
				push!(args_cu, CuArray(arg_el))
			elseif arg_el_type <: CuArray
				# println("CuArray")
				push!(args_jl_ty, Array{eltype(arg_el), ndims(arg_el)})
				push!(args_cu, arg_el)
			else
				# Other element type
			end
		else
			# Other type
			# should not be allowed?
		end
	end

	# Get internal function name
	internal_name = function_name_llvm(func, tuple(args_jl_ty...))
	# Get cuda function object
	cuda_func = CuFunction(md, internal_name)
	# Launch cuda object
	launch(cuda_func, grid, block, tuple(args_cu...), shmem_bytes=shared_bytes)

	# Get results
	index = 1
	for arg in args
		if isa(arg, CuOut) || isa(arg, CuInOut)
			host = to_host(args_cu[index])
			if isa(arg.data, Array)
				copy!(arg.data, host)
			elseif isa(arg.data, CuArray)
				println("Copy to CuArray")
			end
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
