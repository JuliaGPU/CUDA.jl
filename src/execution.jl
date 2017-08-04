# Native execution support

export @cuda, nearest_warpsize, cudaconvert

using Base.Iterators: filter


"""
    cudaconvert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

For `CuArray` objects, a corresponding `CuDeviceArray` object in global space is returned,
which implements GPU-compatible array functionality.
"""
function cudaconvert(x::T) where {T}
    if T.layout == C_NULL || !Base.datatype_pointerfree(T)
        error("don't know how to handle argument of type $T")
    end

    return x
end


# fast lookup of global world age
world_age() = ccall(:jl_get_tls_world_age, UInt, ())

# slow lookup of local method age
function method_age(f, tt)
    for m in Base._methods(f, tt, 1, typemax(UInt))
        return m[3].min_world
    end
    return -1
end


isghosttype(dt) = !dt.mutable && sizeof(dt) == 0


"""
    @cuda (gridDim::CuDim, blockDim::CuDim, [shmem::Int], [stream::CuStream]) func(args...)

High-level interface for calling functions on a GPU, queues a kernel launch on the current
context. The `gridDim` and `blockDim` arguments represent the launch configuration, the
optional `shmem` parameter specifies how much bytes of dynamic shared memory should be
allocated (defaulting to 0), while the optional `stream` parameter indicates on which stream
the launch should be scheduled.

The `func` argument should be a valid Julia function. It will be compiled to a CUDA function
upon first use, and to a certain extent arguments will be converted and managed
automatically (see [`cudaconvert`](@ref)). Finally, a call to `cudacall` is performed,
scheduling the compiled function for execution on the GPU.
"""
macro cuda(config::Expr, callexpr::Expr)
    # sanity checks
    if config.head != :tuple || !(2 <= length(config.args) <= 4)
        throw(ArgumentError("first argument to @cuda should be a tuple (gridDim, blockDim, [shmem], [stream])"))
    end
    if callexpr.head != :call
        throw(ArgumentError("second argument to @cuda should be a function call"))
    end

    # handle optional arguments and forward the call
    # NOTE: we duplicate the CUDAdrv's default values of these arguments,
    #       because the kwarg version of `cudacall` is too slow
    stream = length(config.args)==4 ? esc(pop!(config.args)) : :(CuDefaultStream())
    shmem  = length(config.args)==3 ? esc(pop!(config.args)) : :(0)
    dims = esc(config)
    args = :(cudaconvert.(($(map(esc, callexpr.args)...),)))
    return :(_cuda($dims, $shmem, $stream, $args...))
end

const agecache = Dict{UInt, UInt}()
const compilecache = Dict{UInt, CuFunction}()
@generated function _cuda(dims::Tuple{CuDim, CuDim}, shmem, stream,
                          func::Core.Function, argspec...)
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = argspec

    # filter out ghost arguments
    real_args = map(t->!isghosttype(t), arg_types)
    real_arg_types = map(x->x[2], filter(x->x[1], zip(real_args, arg_types)))
    real_arg_exprs = map(x->x[2], filter(x->x[1], zip(real_args, arg_exprs)))

    precomp_key = hash(tuple(func, arg_types...))  # precomputable part of the keys
    quote
        Base.@_inline_meta

        # look-up the method age
        key1 = hash(($precomp_key, world_age()))
        if haskey(agecache, key1)
            age = agecache[key1]
        else
            age = method_age(func, $arg_types)
            agecache[key1] = age
        end

        # compile the function
        ctx = CuCurrentContext()
        key2 = hash(($precomp_key, age, ctx))
        if haskey(compilecache, key2)
            cuda_fun = compilecache[key2]
        else
            cuda_fun, _ = cufunction(device(ctx), func, Tuple{$arg_types...})
            compilecache[key2] = cuda_fun
        end

        # call the kernel
        Profile.@launch begin
            cudacall(cuda_fun, dims[1], dims[2], shmem, stream,
                     Tuple{$(real_arg_types...)}, $(real_arg_exprs...))
        end
    end
end


"""
Return the nearest number of threads that is a multiple of the warp size of a device:

    nearest_warpsize(dev::CuDevice, threads::Integer)

This is a common requirement, eg. when using shuffle intrinsics.
"""
function nearest_warpsize(dev::CuDevice, threads::Integer)
    ws = CUDAdrv.warpsize(dev)
    return threads + (ws - threads % ws) % ws
end
