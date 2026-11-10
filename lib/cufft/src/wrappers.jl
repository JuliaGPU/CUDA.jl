# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDACore.MAJOR_VERSION),
                          cufftGetProperty(CUDACore.MINOR_VERSION),
                          cufftGetProperty(CUDACore.PATCH_LEVEL))

# cuFFT batches transforms using a single distance, which only covers the dimensions that
# aren't transformed if no transformed dimension lies in between them. Dimensions of size 1
# don't matter either way.
function batchable(sz::Dims, region)
    batch_dims = filter(d -> d ∉ region && sz[d] > 1, 1:length(sz))
    isempty(batch_dims) && return true
    return all(d -> d ∉ region || sz[d] == 1, first(batch_dims):last(batch_dims))
end

# create a cuFFT plan (cufftXtMakePlanMany) that transforms the dimensions `region` of an
# array of size `input_size` and type `input_type` into one of type `output_type`. returns
# the handle and the size of the work area it needs.
function make_plan(output_type::Type{<:cufftNumber}, input_type::Type{<:cufftNumber}, input_size::Dims, region)
    if any(diff(collect(region)) .< 1)
        throw(ArgumentError("FFT region dimensions must be in strictly increasing order; got $region"))
    end
    if any(region .< 1 .|| region .> length(input_size))
        throw(ArgumentError("transformed dims can only refer to valid dimensions"))
    end
    # cuFFT half-precision transforms require all transform dim sizes to be
    # powers of 2 (NVIDIA cuFFT docs, "Half-precision transforms"). Catch this
    # up-front so the user sees a Julia error rather than a bare cuFFT error
    # from inside the wrapper.
    if input_type <: Union{Float16, Complex{Float16}} ||
       output_type <: Union{Float16, Complex{Float16}}
        for d in region
            n = input_size[d]
            ispow2(n) || throw(ArgumentError(
                "cuFFT half-precision transforms require all transform " *
                "dimensions to be powers of 2; got region $region with " *
                "input size $input_size (dim $d has size $n)"))
        end
    end
    nrank = length(region)

    sz = ntuple((d) -> input_size[region[d]], nrank)

    # layouts that aren't batchable are permuted first, see `staging_permutation`
    batchable(input_size, region) ||
        throw(ArgumentError("cuFFT cannot batch over the dimensions around region $region for size $input_size"))
    first_batch_dim = findfirst(d -> d ∉ region && input_size[d] > 1, 1:length(input_size))
    num_batches = prod(input_size) ÷ prod(sz)
    cdims = ntuple((d) -> (d==region[1]) ? div(input_size[d], 2) + 1 : input_size[d], length(input_size))

    # make the plan
    worksize_ref = Ref{Csize_t}()
    rsz = length(sz) > 1 ? reverse(sz) : sz
    if nrank > 3
        throw(ArgumentError("only up to three transform dimensions are allowed in one plan"))
    end

    # initialize the plan handle. the work area is allocated from the memory pool for every
    # execution (see `execute!`), so that idle plans don't hold on to one and concurrent
    # executions of one plan don't share it.
    handle_ref = Ref{cufftHandle}()
    cufftCreate(handle_ref)
    handle = handle_ref[]

    try
        cufftSetAutoAllocation(handle, false)
        execution_type = promote_type(input_type, output_type)
        if region === ntuple(identity, nrank)
            # the transformed dimensions come first, followed by the batch: the default layout
            cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                                C_NULL, 1, 1, convert(cudaDataType, input_type),
                                C_NULL, 1, 1, convert(cudaDataType, output_type),
                                num_batches, worksize_ref, convert(cudaDataType, execution_type))
        else
            # stride between consecutive elements of the innermost transformed dimension
            istride = prod(input_size[1:region[1]-1])

            # distance between consecutive transforms of the batch
            idist = first_batch_dim === nothing ? prod(input_size) : prod(input_size[1:first_batch_dim-1])
            cdist = first_batch_dim === nothing ? prod(cdims) : prod(cdims[1:first_batch_dim-1])

            # embedded storage sizes, in C order. the first entry only has to be at least
            # the outermost transform size; the others are the distances between consecutive
            # elements of each transformed dimension, in units of the stride.
            inembed = Clonglong[max(idist, rsz[1]), (prod(input_size[region[i]:region[i+1]-1]) for i in nrank-1:-1:1)...]
            cnembed = Clonglong[max(cdist, rsz[1]), (prod(cdims[region[i]:region[i+1]-1]) for i in nrank-1:-1:1)...]

            # the complex side of a real transform is laid out like the halved array
            ostride = istride
            onembed, odist = input_type <: Real ? (cnembed, cdist) : (inembed, idist)
            if output_type <: Real
                inembed, idist = cnembed, cdist
            end

            cufftXtMakePlanMany(handle, nrank, Clonglong[rsz...],
                                inembed, istride, idist, convert(cudaDataType, input_type),
                                onembed, ostride, odist, convert(cudaDataType, output_type),
                                num_batches, worksize_ref, convert(cudaDataType, execution_type))
        end
    catch
        cufftDestroy(handle)
        rethrow()
    end

    handle, worksize_ref[]
end


## plan cache

const cufftHandleCacheKey = Tuple{CuContext, Type, Type, Dims, Any}
function handle_ctor((ctx, args...))
    context!(ctx) do
        # make the plan
        handle, _ = make_plan(args...)
        handle
    end
end
function handle_dtor((ctx, args...), handle)
    context!(ctx) do
        cufftDestroy(handle)
    end
end
const idle_handles = HandleCache{cufftHandleCacheKey, cufftHandle}(handle_ctor, handle_dtor)

function get_plan(args...)
    ctx = context()
    return pop!(idle_handles, (ctx, args...))
end
function release_plan(plan)
    push!(idle_handles, plan)
end
