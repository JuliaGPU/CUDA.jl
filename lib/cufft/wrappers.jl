# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDA.MAJOR_VERSION),
                          cufftGetProperty(CUDA.MINOR_VERSION),
                          cufftGetProperty(CUDA.PATCH_LEVEL))

function cufftMakePlan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, xdims::Dims, region)
    if any(diff(collect(region)) .< 1)
        throw(ArgumentError("region must be an increasing sequence"))
    end
    if any(region .< 1 .|| region .> length(xdims))
        throw(ArgumentError("region can only refer to valid dimensions"))
    end
    nrank = length(region)
    sz = [xdims[i] for i in region]
    csz = copy(sz)
    csz[1] = div(sz[1],2) + 1
    batch = prod(xdims) ÷ prod(sz)

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
                            batch, worksize_ref, convert(cudaDataType, execution_type))
    else
        # reduce the array to the final transform direction.
        # This situation will be picked up in the application of the plan later.
        if region[end] != length(xdims)
            # just make a plan for a smaller dimension number
            xdims = xdims[1:region[end]]
            batch = prod(xdims) ÷ prod(sz)
            # throw(ArgumentError("batching dims must be sequential"))
        end

        if nrank==1 || all(diff(collect(region)) .== 1)
            # _stride: successive elements in innermost dimension
            # _dist: distance between first elements of batches
            if region[1] == 1
                istride = 1
                idist = prod(sz)
                cdist = prod(csz)
            else
                istride = prod(xdims[1:region[1]-1])
                idist = 1
                cdist = 1
            end
            inembed = Clonglong[rsz...]
            cnembed = (length(csz) > 1) ? Clonglong[reverse(csz)...] : Clonglong[csz[1]]
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
        else
            # multiple non-sequential transforms
            if any(diff(collect(region)) .< 1)
                cufftDestroy(handle)
                throw(ArgumentError("region must be an increasing sequence"))
            end
            cdims = collect(xdims)
            cdims[region[1]] = div(cdims[region[1]],2)+1

            if region[1] == 1
                istride = 1
                ii=1
                while (ii < nrank) && (region[ii] == region[ii+1]-1)
                    ii += 1
                end
                idist = prod(xdims[1:ii])
                cdist = prod(cdims[1:ii])
                ngaps = 0
            else
                istride = prod(xdims[1:region[1]-1])
                idist = 1
                cdist = 1
                ngaps = 1
            end
            nem = ones(Int,nrank)
            cem = ones(Int,nrank)
            id = 1
            for ii=1:nrank-1
                if region[ii+1] > region[ii]+1
                    ngaps += 1
                end
                while id < region[ii+1]
                    nem[ii] *= xdims[id]
                    cem[ii] *= cdims[id]
                    id += 1
                end
                @assert nem[ii] >= sz[ii]
            end
            if region[end] < length(xdims)
                ngaps += 1
            end
            # CUFFT represents batches by a single stride (_dist)
            # so we must verify that region is consistent with this:
            if ngaps > 1
                cufftDestroy(handle)
                throw(ArgumentError("batch regions must be sequential"))
            end

            inembed = Clonglong[reverse(nem)...]
            cnembed = Clonglong[reverse(cem)...]
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
        end
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
