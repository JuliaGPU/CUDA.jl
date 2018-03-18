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

# fast lookup of global world age
world_age() = ccall(:jl_get_tls_world_age, UInt, ())

# slow lookup of local method age
function method_age(f, tt)::UInt
    for m in Base._methods(f, tt, 1, typemax(UInt))
        return m[3].min_world
    end
    throw(MethodError(f, tt))
end


isghosttype(dt) = dt.isconcretetype && sizeof(dt) == 0


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

The `func` argument should be a valid Julia function. It will be compiled to a CUDA function
upon first use, and to a certain extent arguments will be converted and managed
automatically (see [`cudaconvert`](@ref)). Finally, a call to `cudacall` is performed,
scheduling the compiled function for execution on the GPU.
"""
macro cuda(ex...)
    # sanity checks
    if length(ex) > 0 && ex[1].head == :tuple
        error("The tuple argument to @cuda has been replaced by keywords: `@cuda threads=... fun(args...)`")
    end
    call = ex[end]
    kwargs = ex[1:end-1]
    if call.head != :call
        throw(ArgumentError("second argument to @cuda should be a function call"))
    end

    # decode the call and assign arguments to variables
    f = call.args[1]
    args = call.args[2:end]
    syms = Tuple(gensym() for arg in args)
    arg_no_splat = map(x-> Meta.isexpr(x, :(...)) ? x.args[1] : x, args)
    syms_splat = map(syms, args) do s, arg
         Meta.isexpr(arg, :(...)) ? Expr(:(...), s) : s
    end

    # convert the arguments, and call _cuda while keeping the original arguments alive
    ex = Expr(:block)
    append!(ex.args, :($sym = $(esc(arg))) for (sym,arg) in zip(syms, arg_no_splat))
    push!(ex.args,
        :(GC.@preserve($(syms...),
                       _cuda($(esc(f)), cudaconvert.(($(syms_splat...),))...;
                             $(map(esc, kwargs)...)))))
    return ex
end

const agecache = Dict{UInt, UInt}()
const compilecache = Dict{UInt, CuFunction}()
@generated function _cuda(func::Core.Function, argspec...; kwargs...)
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = argspec

    # split kwargs, only some are dealt with by the compiler
    compile_kwargs, call_kwargs =
        gen_take_kwargs(kwargs, :minthreads, :maxthreads, :blocks_per_sm, :maxregs)

    # filter out ghost arguments
    real_args = map(t->!isghosttype(t), arg_types)
    real_arg_types =               Type[x[1] for x in zip(arg_types, real_args) if x[2]]
    real_arg_exprs = Union{Expr,Symbol}[x[1] for x in zip(arg_exprs, real_args) if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation will fail).
    # alternatively, make CUDAdrv allow `launch` with non-isbits arguments.
    for (i,dt) in enumerate(real_arg_types)
        if !dt.isbitstype
            real_arg_types[i] = Ptr{Void}
            real_arg_exprs[i] = :(convert(Ptr{Void}, UInt(0xDEADBEEF)))
        end
    end

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
        key2 = hash(($precomp_key, age, ctx, ($(compile_kwargs...),)))
        if haskey(compilecache, key2)
            cuda_fun = compilecache[key2]
        else
            cuda_fun, _ = cufunction(device(ctx), func, Tuple{$arg_types...};
                                     $(compile_kwargs...))
            compilecache[key2] = cuda_fun
        end

        # call the kernel
        Profile.@launch begin
            cudacall(cuda_fun, Tuple{$(real_arg_types...)}, $(real_arg_exprs...);
                     $(call_kwargs...))
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
