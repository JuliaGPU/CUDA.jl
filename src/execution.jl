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
cudaconvert(x) = x
cudaconvert(x::Tuple) = cudaconvert.(x)
@generated function cudaconvert(x::NamedTuple)
    Expr(:tuple, (:($f=cudaconvert(x.$f)) for f in fieldnames(x))...)
end

# fast lookup of global world age
world_age() = ccall(:jl_get_tls_world_age, UInt, ())

# slow lookup of local method age
function method_age(f, tt)::UInt
    for m in Base._methods(f, tt, 1, typemax(UInt))
        return m[3].min_world
    end
    throw(MethodError(f, tt))
end

struct Kernel
    ctx::CuContext
    mod::CuModule
    fun::CuFunction
end

# split keyword arguments to `@cuda` into ones affecting the compiler, or the execution
function split_kwargs(kwargs)
    compiler_kws = [:minthreads, :maxthreads, :blocks_per_sm, :maxregs]
    call_kws     = [:blocks, :threads, :shmem, :stream]
    compiler_kwargs = []
    call_kwargs = []
    for kwarg in kwargs
        if Meta.isexpr(kwarg, :(=))
            key,val = kwarg.args
            if isa(key, Symbol)
                if key in compiler_kws
                    push!(compiler_kwargs, kwarg)
                elseif key in call_kws
                    push!(call_kwargs, kwarg)
                else
                    throw(ArgumentError("unknown keyword argument '$key'"))
                end
            else
                throw(ArgumentError("non-symbolic keyword '$key'"))
            end
        else
            throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        end
    end

    return compiler_kwargs, call_kwargs
end

# assign arguments to variables, handle splatting
function assign_args!(code, args)
    # handle splatting
    splats = map(arg -> Meta.isexpr(arg, :(...)), args)
    args = map(args, splats) do arg, splat
        splat ? arg.args[1] : arg
    end

    # assign arguments to variables
    vars = Tuple(gensym() for arg in args)
    map(vars, args) do var,arg
        push!(code.args, :($var = $(esc(arg))))
    end

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    var_exprs = map(vars, args, splats) do var, arg, splat
         splat ? Expr(:(...), var) : var
    end

    return vars, var_exprs
end

"""
    @cuda [kwargs...] func(args...)

High-level interface for calling functions on a GPU, queues a kernel launch on the current
context. The `@cuda` macro should prefix a kernel invocation, with one of the following
arguments in the `kwargs` position:

Affecting the kernel launch:
- threads (defaults to 1)
- blocks (defaults to 1)
- shmem (defaults to 0)
- stream (defaults to the default stream)

Affecting the kernel compilation:
- minthreads: the required number of threads in a thread block.
- maxthreads: the maximum number of threads in a thread block.
- blocks_per_sm: a minimum number of thread blocks to be scheduled on a single
  multiprocessor.
- maxregs: the maximum number of registers to be allocated to a single thread (only
  supported on LLVM 4.0+)

Note that, contrary to with CUDA C, you can invoke the same kernel multiple times with
different compilation parameters. New code will be generated automatically.

The `func` argument should be a valid Julia function. Its return values will be ignored, by
means of a wrapper. The function will be compiled to a CUDA function upon first use, and to
a certain extent arguments will be converted and managed automatically (see
[`cudaconvert`](@ref)). Finally, a call to `cudacall` is performed, scheduling the compiled
function for execution on the GPU.
"""
macro cuda(ex...)
    # destructure the `@cuda` expression
    if length(ex) > 0 && ex[1].head == :tuple
        error("The tuple argument to @cuda has been replaced by keywords: `@cuda threads=... fun(args...)`")
    end
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the kernel call
    if call.head != :call
        throw(ArgumentError("second argument to @cuda should be a function call"))
    end
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    compiler_kwargs, call_kwargs = split_kwargs(kwargs)
    vars, var_exprs = assign_args!(code, args)

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    @gensym kernel cuda_args # FIXME: why doesn't `local` work
    push!(code.args,
        quote
            GC.@preserve $(vars...) begin
                $cuda_args = cudaconvert.(($(var_exprs...),))
                $kernel = compile_function($(esc(f)),
                                           $cuda_args...; $(map(esc, compiler_kwargs)...))
                call_kernel($kernel, $(esc(f)),
                            $cuda_args...; $(map(esc, call_kwargs)...))
            end
         end)
    return code
end

const agecache = Dict{UInt, UInt}()
const compilecache = Dict{UInt, Kernel}()

@generated function compile_function(f::Core.Function, args...; kwargs...)
    # we're in a generated function, so `args` are really types.
    # destructure into more appropriately-named variables
    t = args
    sig = (f, t...)
    args = (:f, (:( args[$i] ) for i in 1:length(args))...)

    # finalize types
    tt = Base.to_tuple_type(t)

    precomp_key = hash(sig)  # precomputable part of the keys
    quote
        Base.@_inline_meta

        CUDAnative.maybe_initialize("@cuda")

        # look-up the method age
        key1 = hash(($precomp_key, world_age()))
        if haskey(agecache, key1)
            age = agecache[key1]
        else
            age = method_age(f, $t)
            agecache[key1] = age
        end

        # compile the function
        ctx = CuCurrentContext()
        key2 = hash(($precomp_key, age, ctx, kwargs))
        if !haskey(compilecache, key2)
            fun, mod = cufunction(device(ctx), f, $tt; kwargs...)
            kernel = Kernel(ctx, mod, fun)
            compilecache[key2] = kernel
        end
        kernel = compilecache[key2]
    end
end

@generated function call_kernel(kernel, f::Core.Function, args...; call_kwargs...)
    # we're in a generated function, so `args` are really types.
    # destructure into more appropriately-named variables
    t = args
    sig = (f, t...)
    args = (:f, (:( args[$i] ) for i in 1:length(args))...)

    # filter out ghost arguments that shouldn't be passed
    to_pass = map(!isghosttype, sig)
    call_t =                  Type[x[1] for x in zip(sig,  to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass) if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    # alternatively, make CUDAdrv allow `launch` with non-isbits arguments.
    for (i,dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Cvoid}
            call_args[i] = :C_NULL
        end
    end

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        Base.@_inline_meta

        # call the kernel
        cudacall(kernel.fun, $call_tt, $(call_args...); call_kwargs...)
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
