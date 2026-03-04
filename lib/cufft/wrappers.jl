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
    cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, xdims::Dims, region)

low level interface to the CUDA library CuFFT for the function cufftXtMakePlanMany

# Parameters:
* `output_type `: type of the input array
* `input_type`: type of the output array
* `xdim`: size of the array to transform in units of the type
* `region`: dimensions of the array to transform
"""
function cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, xdims::Dims, region)
    if any(diff(collect(region)) .!= 1)
        throw(ArgumentError("transformed dims must be an increasing sequence"))
    end
    if any(region .< 1 .|| region .> length(xdims))
        throw(ArgumentError("transformed dims can only refer to valid dimensions"))
    end
    nrank = length(region)
    sz = [xdims[i] for i in region]
    csz = copy(sz)
    csz[1] = div(sz[1],2) + 1
    # all sizes which are not part of the dimensions specified by region are batch dimensions.
    batch = prod(xdims) ÷ prod(sz) 
    cdims = ntuple((d) -> (d==1) ? div(xdims[1],2) + 1 : xdims[1], length(xdims))
    inembed = Clonglong[reverse(xdims[region[1]:region[end]])..., 1]
    cnembed = Clonglong[reverse(cdims[region[1]:region[end]])..., 1]

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
        # @show "contiguous dims, no batching"
        # handle simple case, transforming the first nrank dimensions, ... simply! (for robustness)
        # arguments are: plan, rank, transform-sizes, inembed, istride, idist, itype, onembed, ostride, odist, otype, batch
        execution_type = promote_type(input_type, output_type)
        cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                            C_NULL, 1, 1, convert(cudaDataType, input_type),
                            C_NULL, 1, 1, convert(cudaDataType, output_type),
                            batch, worksize_ref, convert(cudaDataType, execution_type))
    else
        # @show "internal batching needed"

        # reduce the array to the final transform direction.
        # This situation will be picked up in the application of the plan later.
        internal_dims, external_dims = get_batch_dims(region, xdims)
        # plan only for the internal dims and the external dims will be handled later

        extra_stride = 1
        # front is internally batched. Make a plan ending at the last transform dim
        if (internal_dims[1] != 1)
            extra_stride = prod(xdims[1:region[1]-1])
            # batch stride:
            idist = prod(xdims[1:region[end]])
            cdist = prod(csz)
            # first fft-dimension stride:
            istride = prod(xdims[1:region[1]-1])
            xdims = xdims[region[1]:end]
        else
            # just make a plan for a smaller dimension number
            # pretend the array is smaller (end is handled externally)
            # back is internal and front part will be executed via a for loop.
            # ignore all front dimensions for the plan
            extra_stride = 1
            # batch stride:
            idist = 1
            cdist = 1
            # first fft-dimension stride:
            istride = prod(xdims[1:region[1]-1])
            xdims = xdims[1:region[end]]
        end
        # internal number of batches are at the front: remaining size divided by transformed sizes
        batch = prod(xdims) ÷ prod(sz)

        # 2D: input[ b * idist + (x * inembed[1] + y) * istride ]
        # 3D: input[ b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride ]
        # first transform dimension stride, assuming internal batching over outer dimensions
        # batching stride (datatype size dependent)
        # cdist = prod(csz)*extra_stride

        ostride = istride
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
        # end
        execution_type = promote_type(input_type, output_type)

        res = cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                                  inembed, istride, idist, convert(cudaDataType, input_type),
                                  onembed, ostride, odist, convert(cudaDataType, output_type),
                                  batch, worksize_ref, convert(cudaDataType, execution_type))
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
    context!(ctx; skip_destroyed=true) do
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
