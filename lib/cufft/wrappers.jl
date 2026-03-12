# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
	value_ref = Ref{Cint}()
	cufftGetProperty(property, value_ref)
	value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDA.MAJOR_VERSION),
	cufftGetProperty(CUDA.MINOR_VERSION),
	cufftGetProperty(CUDA.PATCH_LEVEL))

"""
	cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)

low level interface to the CUDA library CuFFT for the function cufftXtMakePlanMany

# Parameters:
* `output_type `: type of the input array
* `input_type`: type of the output array
* `input_size`: size of the array to transform in units of the type
* `region`: dimensions of the array to transform
"""
function cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)
	if any(diff(collect(region)) .< 1)
		throw(ArgumentError("transformed dims for rfft-type transforms must be an increasing sequence"))
	end
	if any(region .< 1 .|| region .> length(input_size))
		throw(ArgumentError("transformed dims can only refer to valid dimensions"))
	end
	nrank = length(region)

	sz = ntuple((d) -> input_size[region[d]], nrank)
	csz = ntuple((d) -> (d==1) ? div(input_size[region[d]], 2) + 1 : input_size[region[d]], nrank)

	# all sizes which are not part of the dimensions specified by region are batch dimensions.
	num_internal_batches = prod(input_size) ÷ prod(sz)
	cdims = ntuple((d) -> (d==region[1]) ? div(input_size[d], 2) + 1 : input_size[d], length(input_size))

	# make the plan
	worksize_ref = Ref{Csize_t}()
	rsz = length(sz) > 1 ? rsz = reverse(sz) : sz
	if nrank > 3
		throw(ArgumentError("only up to three transform dimensions are allowed in one plan"))
	end

	# initialize the plan handle
	handle_ref = Ref{cufftHandle}()
	cufftCreate(handle_ref)
	handle = handle_ref[]

	if (region...,) == ((1:nrank)...,)
		# handle simple case, transforming the first nrank dimensions, ... simply! (for robustness)
		# arguments are: plan, rank, transform-sizes, inembed, istride, idist, itype, onembed, ostride, odist, otype, batch
		execution_type = promote_type(input_type, output_type)
		cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
			C_NULL, 1, 1, convert(cudaDataType, input_type),
			C_NULL, 1, 1, convert(cudaDataType, output_type),
			num_internal_batches, worksize_ref, convert(cudaDataType, execution_type))
	else
		# reduce the array to the final transform direction.
		# This situation will be picked up in the application of the plan later.
		internal_batch_dims, external_batch_dims = get_batch_dims(region, input_size)
		# plan only for the internal dims and the external dims will be handled later

		# the internal batching is over the largest consecutive batch indices.
		# Since they are consecutive they can all be batched by a single batch_stride "i_dist".
		idist = prod((1, input_size...)[1:internal_batch_dims[1]])

		cdist = prod((1, cdims...)[1:internal_batch_dims[1]])
		istride = prod((1, input_size...)[1:region[1]])

		# inembed are the products of the sizes between the transform directions 
		all_strides = cumprod((1, input_size...))
		if (nrank == 1)
			# For 1D transforms, inembed[0] must be >= n[0] (the transform size)
			# Using idist gives the embedded storage size which satisfies this requirement
			inembed = Clonglong[idist,]
			cnembed = Clonglong[cdist,]
		elseif (nrank == 2)
			# inembed[0] must be >= n[0] (the larger transform dim in C-order)
			# Using idist ensures sufficient storage is indicated
			inembed = Clonglong[idist, prod(input_size[region[1]:(region[2]-1)])]
			cnembed = Clonglong[cdist, prod(cdims[region[1]:(region[2]-1)])]
		elseif (nrank == 3)
			inembed = Clonglong[idist, prod(input_size[region[2]:(region[3]-1)]), prod(input_size[region[1]:(region[2]-1)])]
			cnembed = Clonglong[cdist, prod(cdims[region[2]:(region[3]-1)]), prod(cdims[region[1]:(region[2]-1)])]
		end

		# internal number of batches are product of the sizes at the internal_batch_dims
		num_internal_batches = prod(input_size[[internal_batch_dims...]])

		# in C-style notation:
		# 2D: input[ b * idist + (x * inembed[1] + y) * istride ]
		# 3D: input[ b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride ]
		# first transform dimension stride, assuming internal batching over outer dimensions
		# batching stride (datatype size dependent)

		ostride = istride
		# if input_type is real, the output will be half complex, so the output strides are modified
		if input_type <: Real
			odist = cdist
			onembed = cnembed
		else
			odist = idist
			onembed = inembed
		end
		if output_type <: Real
			idist = cdist
			inembed = cnembed
		end

		execution_type = promote_type(input_type, output_type)

		res = cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
			inembed, istride, idist, convert(cudaDataType, input_type),
			onembed, ostride, odist, convert(cudaDataType, output_type),
			num_internal_batches, worksize_ref, convert(cudaDataType, execution_type))
	end

	handle, worksize_ref[]
end


## plan cache

const cufftHandleCacheKey = Tuple{CuContext, Type, Type, Dims, Any}
function handle_ctor((ctx, args...))
	context!(ctx) do
		# make the plan
		handle, worksize = cufftMakePlan(args...)

		# NOTE: we currently do not use the worksize to allocate our own workarea,
		#       instead relying on the automatic allocation strategy.
		handle
	end
end
function handle_dtor((ctx, args...), handle)
	context!(ctx; skip_destroyed = true) do
		cufftDestroy(handle)
	end
end
const idle_handles = HandleCache{cufftHandleCacheKey, cufftHandle}(handle_ctor, handle_dtor)

function cufftGetPlan(args...)
	ctx = context()
	handle = pop!(idle_handles, (ctx, args...))

	cufftSetStream(handle, stream())

	return handle
end
function cufftReleasePlan(plan)
	push!(idle_handles, plan)
end
