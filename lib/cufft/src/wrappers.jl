# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDACore.MAJOR_VERSION),
                          cufftGetProperty(CUDACore.MINOR_VERSION),
                          cufftGetProperty(CUDACore.PATCH_LEVEL))

"""
    cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)

low level interface to the CUDA library CuFFT for the function cufftXtMakePlanMany

# Parameters:
* `output_type`: type of the output array
* `input_type`: type of the input array
* `input_size`: size of the array to transform in units of the type
* `region`: dimensions of the array to transform
"""
function cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)
    if any(diff(collect(region)) .< 1)
        throw(ArgumentError("FFT region dimensions must be in strictly increasing order; got $region"))
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
    rsz = length(sz) > 1 ? reverse(sz) : sz
    if nrank > 3
        throw(ArgumentError("only up to three transform dimensions are allowed in one plan"))
    end

    # initialize the plan handle
    handle_ref = Ref{cufftHandle}()
    cufftCreate(handle_ref)
    handle = handle_ref[]

    if region === ntuple(identity, nrank)
        # handle simple case, transforming the first nrank dimensions, ... simply! (for robustness)
        # arguments are: plan, rank, transform-sizes, inembed, istride, idist, itype, onembed, ostride, odist, otype, batch
        execution_type = promote_type(input_type, output_type)
        cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                            C_NULL, 1, 1, convert(cudaDataType, input_type),
                            C_NULL, 1, 1, convert(cudaDataType, output_type),
                            num_internal_batches, worksize_ref, convert(cudaDataType, execution_type))
    else
        # reduce the array to the final transform direction.
        # This situation will be picked up in the application of the plan later 
        # so the plan needs to only include internal dims.
        internal_batch_dims, external_batch_dims = get_batch_dims(region, input_size)

        # Stride between consecutive elements in the innermost transform dim
        istride = prod(input_size[1:region[1]-1])
        # The internal batching is over the largest consecutive batch indices.
        # Since they are consecutive they can all be batched by a single batch_stride "i_dist".
        # Distance between consecutive internal batches
        idist = prod(input_size[1:internal_batch_dims[1]-1])
        cdist = prod(cdims[1:internal_batch_dims[1]-1])

        # Embedded storage sizes (C-order): cuFFT requires the first entry >= n[0]
        # (the outermost transform size, which in C-order is rsz[1]). Use idist when
        # it already satisfies this; otherwise pad up to n[0].
        # Remaining entries are products of sizes between consecutive transform dims.
        inembed = Clonglong[max(idist, rsz[1]), (prod(input_size[region[i]:region[i+1]-1]) for i in nrank-1:-1:1)...]
        cnembed = Clonglong[max(cdist, rsz[1]), (prod(cdims[region[i]:region[i+1]-1]) for i in nrank-1:-1:1)...]

        num_internal_batches = prod(input_size[collect(internal_batch_dims)])

        ostride = istride
        # R2C: output uses half-complex layout; C2R: input uses half-complex layout
        onembed, odist = input_type <: Real ? (cnembed, cdist) : (inembed, idist)
        if output_type <: Real
            inembed, idist = cnembed, cdist
        end

        execution_type = promote_type(input_type, output_type)
        cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
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
