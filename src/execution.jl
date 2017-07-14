# Native execution support

export @cuda, nearest_warpsize

using Base.Iterators: filter


#
# Auxiliary
#

# Determine which type to pre-convert objects to for use on a CUDA device.
#
# The resulting object type will be used as a starting point to determine the final argument
# types. This is different from `cconvert` in that we don't know which type to convert to.
function convert_type(t)
    # NOTE: this conversion was originally intended to be a user-extensible interface,
    #       a la cconvert (look for cudaconvert in f1e592e61d6898869b918331e3e625292f4c8cab).
    #
    #       however, the generated function behind @cuda isn't allowed to call overloaded
    #       functions (only pure ones), and also won't be able to see functions defined
    #       after the generated function's body (see JuliaLang/julia#19942).

    # Pointer handling
    if t <: DevicePtr
        return Ptr{t.parameters...}
    elseif t <: Ptr
        throw(InexactError())
    end

    # Array types
    if t <: CuArray
        return CuDeviceArray{t.parameters...}
    end

    return t
end

# Convert the arguments to a kernel function to their CUDA representation, and figure out
# what types to specialize the kernel function for.
function convert_arguments(args, types)
    argtypes = DataType[types...]
    argexprs = Union{Expr,Symbol}[args...]

    # convert types to their CUDA representation
    for i in 1:length(argexprs)
        t = argtypes[i]
        ct = convert_type(t)
        if ct != t
            argtypes[i] = ct
            if ct <: Ptr
                argexprs[i] = :( Base.unsafe_convert($ct, $(argexprs[i])) )
            else
                argexprs[i] = :( convert($ct, $(argexprs[i])) )
            end
        end
    end

    # NOTE: DevicePtr's should have disappeared after this point

    for argtype in argtypes
        if argtype.layout == C_NULL || !Base.datatype_pointerfree(argtype)
            error("don't know how to handle argument of type $argtype")
        end
    end

    return argexprs, argtypes
end

function emit_cudacall(func, dims, shmem, stream, types, args)
    # TODO: can we handle non-isbits types?
    all(t -> isbits(t) && sizeof(t) > 0, types) ||
        error("can only pass bitstypes of size > 0 to CUDA kernels")

    return quote
        Profile.@launch begin
            cudacall($func, $dims[1], $dims[2], $shmem, $stream, Tuple{$(types...)}, $(args...))
        end
    end
end

world_age() = ccall(:jl_get_tls_world_age, UInt, ())

isghosttype(dt) = !dt.mutable && sizeof(dt) == 0


#
# @cuda macro
#

"""
    @cuda (gridDim::CuDim, blockDim::CuDim, [shmem::Int], [stream::CuStream]) func(args...)

High-level interface for calling functions on a GPU, queues a kernel launch on the current
context. The `gridDim` and `blockDim` arguments represent the launch configuration, the
optional `shmem` parameter specifies how much bytes of dynamic shared memory should be
allocated (defaulting to 0), while the optional `stream` parameter indicates on which stream
the launch should be scheduled.

The `func` argument should be a valid Julia function. It will be compiled to a CUDA function
upon first use, and to a certain extent arguments will be converted and managed
automatically. Finally, a call to `cudacall` is performed, scheduling the compiled function
for execution on the GPU.
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
    return :(generated_cuda($dims, $shmem, $stream, $(map(esc, callexpr.args)...)))
end

# Compile and execute a CUDA kernel from a Julia function
const compilecache = Dict{UInt, CuFunction}()
@generated function generated_cuda{F<:Core.Function,N}(dims::Tuple{CuDim, CuDim}, shmem, stream,
                                                       func::F, args::Vararg{Any,N})
    arg_exprs = [:( args[$i] ) for i in 1:N]
    arg_exprs, arg_types = convert_arguments(arg_exprs, args)

    # compile the function, if necessary
    # TODO: we currently recompile if _any_ world change is detected.
    #       this is obviously much to coarse, and we should figure out a way to efficiently
    #       query the kernel method's age, and/or put that query in a wrapper method with a
    #       back-edge from the kernel method to make the check completely free.
    @gensym cuda_fun
    precomp_key = hash(tuple(func, arg_types...))  # precomputable part of the key
    kernel_compilation = quote
        ctx = CuCurrentContext()
        key = hash(($precomp_key, ctx, world_age()))
        if (haskey(compilecache, key))
            $cuda_fun = compilecache[key]
        else
            $cuda_fun, _ = cufunction(device(ctx), func, $arg_types)
            compilecache[key] = $cuda_fun
        end
    end

    # filter out non-concrete args
    concrete = map(t->!isghosttype(t), arg_types)
    arg_types = map(x->x[2], filter(x->x[1], zip(concrete, arg_types)))
    arg_exprs = map(x->x[2], filter(x->x[1], zip(concrete, arg_exprs)))

    kernel_call = emit_cudacall(cuda_fun, :(dims), :(shmem), :(stream),
                                arg_types, arg_exprs)

    quote
        Base.@_inline_meta
        $kernel_compilation
        $kernel_call
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
