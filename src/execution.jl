# Native execution support

export @cuda, cudaconvert, cufunction, nearest_warpsize


## kernel object and query functions

struct Kernel{F}
    ctx::CuContext
    mod::CuModule
    fun::CuFunction
end

"""
    version(k::Kernel)

Queries the PTX and SM versions a kernel was compiled for.
Returns a named tuple.
"""
function version(k::Kernel)
    attr = attributes(k.fun)
    binary_ver = VersionNumber(divrem(attr[CUDAdrv.FUNC_ATTRIBUTE_BINARY_VERSION],10)...)
    ptx_ver = VersionNumber(divrem(attr[CUDAdrv.FUNC_ATTRIBUTE_PTX_VERSION],10)...)
    return (ptx=ptx_ver, binary=binary_ver)
end

"""
    memory(k::Kernel)

Queries the local, shared and constant memory usage of a compiled kernel in bytes.
Returns a named tuple.
"""
function memory(k::Kernel)
    attr = attributes(k.fun)
    local_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
    shared_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
    constant_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_CONST_SIZE_BYTES]
    return (:local=>local_mem, shared=shared_mem, constant=constant_mem)
end

"""
    registers(k::Kernel)

Queries the register usage of a kernel.
"""
function registers(k::Kernel)
    attr = attributes(k.fun)
    return attr[CUDAdrv.FUNC_ATTRIBUTE_NUM_REGS]
end

"""
    maxthreads(k::Kernel)

Queries the maximum amount of threads a kernel can use in a single block.
"""
function maxthreads(k::Kernel)
    attr = attributes(k.fun)
    return attr[CUDAdrv.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
end


## helper functions

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

    # convert the arguments, compile the function and call the kernel
    # while keeping the original arguments alive
    var_exprs = map(vars, args, splats) do var, arg, splat
         splat ? Expr(:(...), var) : var
    end

    return vars, var_exprs
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


## high-level @cuda interface

"""
    @cuda [kwargs...] func(args...)

High-level interface for executing code on a GPU. The `@cuda` macro should prefix a call,
with `func` a callable function or object that should return nothing. It will be compiled to
a CUDA function upon first use, and to a certain extent arguments will be converted and
managed automatically (see [`cudaconvert`](@ref)). Finally, a call to `CUDAdrv.cudacall` is
performed, scheduling a kernel launch on the current CUDA context.


There are certain keyword arguments that influence kernel compilation and calling.

Affecting the kernel compilation:
- minthreads: the required number of threads in a thread block.
- maxthreads: the maximum number of threads in a thread block.
- blocks_per_sm: a minimum number of thread blocks to be scheduled on a single
  multiprocessor.
- maxregs: the maximum number of registers to be allocated to a single thread (only
  supported on LLVM 4.0+)

Affecting the kernel call:
- threads (defaults to 1)
- blocks (defaults to 1)
- shmem (defaults to 0)
- stream (defaults to the default stream)

Note that, contrary to with CUDA C, you can invoke the same kernel multiple times with
different compilation parameters. New code will be generated automatically.


The underlying operations (argument conversion, kernel compilation, kernel call) can be
performed explicitly when more control is needed, e.g. to reflect on the resource usage of a
kernel to determine the launch configuration:

    args = ...
    GC.@preserve args begin
        kernel_args = cudaconvert.(args)
        kernel = CUDAnative.cufunction(f, kernel_args; compilation_kwargs)
        kernel(kernel_args...; launch_kwargs)
    end
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
    @gensym kernel kernel_args # FIXME: why doesn't `local` work
    push!(code.args,
        quote
            GC.@preserve $(vars...) begin
                $kernel_args = cudaconvert.(($(var_exprs...),))
                $kernel = cufunction($(esc(f)), $kernel_args...;
                                           $(map(esc, compiler_kwargs)...))
                $kernel($kernel_args...; $(map(esc, call_kwargs)...))
            end
         end)
    return code
end


## APIs for manual compilation

"""
    cudaconvert(x)

Convert values to a representation that is GPU compatible.

For more information, refer to the documentation of the high-level [`@cuda`](@ref)
interface.

By default, CUDAnative does only provide a minimal set of conversions for elementary types
such as tuples. If you need your type to convert before execution on a GPU, be sure to add
methods to this function.

For the time being, conversions for `CUDAdrv.CuArray` objects are also provided, returning a
corresponding `CuDeviceArray` object in global memory. This will be deprecated in favor of
functionality from the CuArrays.jl package.
"""
cudaconvert(x) = x
cudaconvert(x::Tuple) = cudaconvert.(x)
@generated function cudaconvert(x::NamedTuple)
    Expr(:tuple, (:($f=cudaconvert(x.$f)) for f in fieldnames(x))...)
end

const agecache = Dict{UInt, UInt}()
const compilecache = Dict{UInt, Kernel}()

"""
    cufunction(f, args...; kwargs...)

Compile a function invocation for the currently-active GPU, returning a callable kernel
object.

For more information, and a list of supported keyword arguments, refer to the documentation
of the high-level [`@cuda`](@ref) interface. If you need an even lower-level interface, use
[`compile`](@ref).
"""
@generated function cufunction(f::Core.Function, args...; kwargs...)
    # we're in a generated function, so `args` are really types.
    # destructure into more appropriately-named variables
    t = args
    sig = (f, t...)
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
            fun, mod = compile(device(ctx), f, $tt; kwargs...)
            kernel = Kernel{f}(ctx, mod, fun)
            compilecache[key2] = kernel
        end
        kernel = compilecache[key2]

        @debug begin
            ver = version(kernel)
            mem = memory(kernel)
            reg = registers(kernel)
            """Compiled $f to PTX $(ver.ptx) for SM $(ver.binary) using $reg registers.
               Memory usage: $(mem.local) B local, $(mem.shared) B shared, $(mem.constant) B constant"""
        end

        return kernel
    end
end

"""
    (::kernel)(args...; kwargs...)

Schedule a call to a compiled kernel, passing GPU-compatible arguments in `args`.
This is a low-level interface, see [`@cuda`](@ref) for more information.
"""
@generated function (kernel::Kernel{F})(args...; call_kwargs...) where F
    # we're in a generated function, so `args` are really types.
    # destructure into more appropriately-named variables
    t = args
    sig = (typeof(F), t...)
    args = (:F, (:( args[$i] ) for i in 1:length(args))...)

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

        cudacall(kernel.fun, $call_tt, $(call_args...); call_kwargs...)
    end
end


## other

"""
    nearest_warpsize(dev::CuDevice, threads::Integer)

Return the nearest number of threads that is a multiple of the warp size of a device.

This is a common requirement, eg. when using shuffle intrinsics.
"""
function nearest_warpsize(dev::CuDevice, threads::Integer)
    ws = CUDAdrv.warpsize(dev)
    return threads + (ws - threads % ws) % ws
end
