# Native execution support

export @cuda, cudaconvert, cufunction, dynamic_cufunction, nextwarp, prevwarp


## high-level @cuda interface

const MACRO_KWARGS = [:dynamic, :launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :minthreads, :maxthreads, :blocks_per_sm, :maxregs, :fastmath, :cap, :ptx]
const LAUNCH_KWARGS = [:cooperative, :blocks, :threads, :shmem, :stream]


"""
    @cuda [kwargs...] func(args...)

High-level interface for executing code on a GPU. The `@cuda` macro should prefix a call,
with `func` a callable function or object that should return nothing. It will be compiled to
a CUDA function upon first use, and to a certain extent arguments will be converted and
managed automatically using `cudaconvert`. Finally, a call to `cudacall` is
performed, scheduling a kernel launch on the current CUDA context.

Several keyword arguments are supported that influence the behavior of `@cuda`.
- `launch`: whether to launch this kernel, defaults to `true`. If `false` the returned
  kernel object should be launched by calling it and passing arguments again.
- `dynamic`: use dynamic parallelism to launch device-side kernels, defaults to `false`.
- arguments that influence kernel compilation: see [`cufunction`](@ref) and
  [`dynamic_cufunction`](@ref)
- arguments that influence kernel launch: see [`CUDA.HostKernel`](@ref) and
  [`CUDA.DeviceKernel`](@ref)
"""
macro cuda(ex...)
    # destructure the `@cuda` expression
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @cuda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    dynamic = false
    launch = true
    for kwarg in macro_kwargs
        key::Symbol, val = kwarg.args
        if key === :dynamic
            isa(val, Bool) || throw(ArgumentError("`dynamic` keyword argument to @cuda should be a constant value"))
            dynamic = val::Bool
        elseif key === :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @cuda should be a constant value"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@cuda with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel
    if dynamic
        # FIXME: we could probably somehow support kwargs with constant values by either
        #        saving them in a global Dict here, or trying to pick them up from the Julia
        #        IR when processing the dynamic parallelism marker
        isempty(compiler_kwargs) || error("@cuda dynamic parallelism does not support compiler keyword arguments")

        # dynamic, device-side kernel launch
        push!(code.args,
            quote
                # we're in kernel land already, so no need to cudaconvert arguments
                $kernel_args = ($(var_exprs...),)
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $dynamic_cufunction($f, $kernel_tt)
                if $launch
                    $kernel($kernel_args...; $(call_kwargs...))
                end
                $kernel
             end)
    else
        # regular, host-side kernel launch
        #
        # convert the function, its arguments, call the compiler and launch the kernel
        # while keeping the original arguments alive
        push!(code.args,
            quote
                $f_var = $f
                GC.@preserve $(vars...) $f_var begin
                    $kernel_f = $cudaconvert($f_var)
                    $kernel_args = map($cudaconvert, ($(var_exprs...),))
                    $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                    $kernel = $cufunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                    if $launch
                        $kernel($(var_exprs...); $(call_kwargs...))
                    end
                    $kernel
                end
             end)
    end

    # wrap everything in a let block so that temporary variables don't leak in the REPL
    return esc(quote
        let
            $code
        end
    end)
end


## host to device value conversion

struct KernelAdaptor end

# convert CUDA host pointers to device pointers
# TODO: use ordinary ptr?
Adapt.adapt_storage(to::KernelAdaptor, p::CuPtr{T}) where {T} =
    reinterpret(LLVMPtr{T,AS.Generic}, p)

# convert CUDA host arrays to device arrays
function Adapt.adapt_storage(::KernelAdaptor, xs::DenseCuArray{T,N}) where {T,N}
  # prefetch unified memory as we're likely to use it on the GPU
  # TODO: make this configurable?
  if is_unified(xs)
    # XXX: use convert to pointer and/or prefect(CuArray)
    mem = xs.data[].mem::UnifiedMemory

    can_prefetch = sizeof(xs) > 0
    ## prefetching isn't supported during stream capture
    can_prefetch &= !is_capturing()
    ## we can only prefetch pageable memory
    can_prefetch &= !__pinned(convert(Ptr{T}, mem), mem.ctx)
    ## pageable memory needs to be accessible concurrently
    can_prefetch &= attribute(device(), DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) == 1

    if can_prefetch
        # TODO: `view` on buffers?
        subbuf = UnifiedMemory(mem.ctx, pointer(xs), sizeof(xs))
        prefetch(subbuf)
    end
  end

  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)
end

# Base.RefValue isn't GPU compatible, so provide a compatible alternative.
# Note that it isn't safe to use unified or heterogeneous memory to support a
# mutable Ref, because there's no guarantee that the memory would be kept alive
# long enough (especially with broadcast using ephemeral Refs for scalar args).
struct CuRefValue{T} <: Ref{T}
    val::T
end
Base.getindex(r::CuRefValue{T}) where T = r.val
Adapt.adapt_structure(to::KernelAdaptor, ref::Base.RefValue) =
    CuRefValue(adapt(to, ref[]))

# broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
# avoid that by using a special kind of ref that knows about the boxed type.
struct CuRefType{T} <: Ref{DataType} end
Base.getindex(r::CuRefType{T}) where T = T
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue{<:Union{DataType,Type}}) =
    CuRefType{r[]}()

# case where type is the function being broadcasted
Adapt.adapt_structure(to::KernelAdaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

"""
    cudaconvert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `CUDA.KernelAdaptor` type.
"""
cudaconvert(arg) = adapt(KernelAdaptor(), arg)


## abstract kernel functionality

abstract type AbstractKernel{F,TT} end

# FIXME: there doesn't seem to be a way to access the documentation for the call-syntax,
#        so attach it to the type -- https://github.com/JuliaDocs/Documenter.jl/issues/558

"""
    (::HostKernel)(args...; kwargs...)
    (::DeviceKernel)(args...; kwargs...)

Low-level interface to call a compiled kernel, passing GPU-compatible arguments
in `args`. For a higher-level interface, use [`@cuda`](@ref).

A `HostKernel` is callable on the host, and a `DeviceKernel` is callable on the
device (created by `@cuda` with `dynamic=true`).

The following keyword arguments are supported:
- `threads` (default: `1`): Number of threads per block, or a 1-, 2- or 3-tuple of dimensions
  (e.g. `threads=(32, 32)` for a 2D block of 32×32 threads).
  Use [`threadIdx()`](@ref) and [`blockDim()`](@ref) to query from within the kernel.
- `blocks` (default: `1`): Number of thread blocks to launch, or a 1-, 2- or 3-tuple of
  dimensions (e.g. `blocks=(2, 4, 2)` for a 3D grid of blocks).
  Use [`blockIdx()`](@ref) and [`gridDim()`](@ref) to query from within the kernel.
- `shmem`(default: `0`): Amount of dynamic shared memory in bytes to allocate per thread block;
  used by [`CuDynamicSharedArray`](@ref).
- `stream` (default: [`stream()`](@ref)): [`CuStream`](@ref) to launch the kernel on.
- `cooperative` (default: `false`): whether to launch a cooperative kernel that supports
  grid synchronization (see [`CG.this_grid`](@ref) and [`CG.sync`](@ref)).
  Note that this requires care wrt. the number of blocks launched.
"""
AbstractKernel

function Base.show(io::IO, k::AbstractKernel{F,TT}) where {F,TT}
    print(io, "CUDA.$(nameof(typeof(k)))($(k.f))")
end
function Base.show(io::IO, ::MIME"text/plain", k::AbstractKernel{F,TT}) where {F,TT}
    print(io, "CUDA.$(nameof(typeof(k))) for $(k.f)($(join(TT.parameters, ", ")))")
end

@inline @generated function call(kernel::AbstractKernel{F,TT}, args...; call_kwargs...) where {F,TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:( args[$i] ) for i in 1:length(args))...)

    # filter out arguments that shouldn't be passed
    predicate = dt -> isghosttype(dt) || Core.Compiler.isconstType(dt)
    to_pass = map(!predicate, sig.parameters)
    call_t =                  Type[x[1] for x in zip(sig.parameters,  to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass)            if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    # alternatively, make it possible to `launch` with non-isbits arguments.
    for (i,dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Any}
            call_args[i] = :C_NULL
        end
    end

    # add the kernel state, passing an instance with a unique seed
    pushfirst!(call_t, KernelState)
    pushfirst!(call_args, :(KernelState(kernel.state.exception_info, make_seed(kernel))))

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        cudacall(kernel.fun, $call_tt, $(call_args...); call_kwargs...)
    end
end


## host-side kernels

# XXX: storing the function instance, but not the arguments, is inconsistent.
#      either store the instance and args, making this object directly callable,
#      or store neither and cache it when getting it directly from GPUCompiler.

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::CuFunction
    state::KernelState
end

@doc (@doc AbstractKernel) HostKernel

"""
    version(k::HostKernel)

Queries the PTX and SM versions a kernel was compiled for.
Returns a named tuple.
"""
function version(k::HostKernel)
    attr = attributes(k.fun)
    binary_ver = VersionNumber(divrem(attr[FUNC_ATTRIBUTE_BINARY_VERSION],10)...)
    ptx_ver = VersionNumber(divrem(attr[FUNC_ATTRIBUTE_PTX_VERSION],10)...)
    return (ptx=ptx_ver, binary=binary_ver)
end

"""
    memory(k::HostKernel)

Queries the local, shared and constant memory usage of a compiled kernel in bytes.
Returns a named tuple.
"""
function memory(k::HostKernel)
    attr = attributes(k.fun)
    local_mem = attr[FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
    shared_mem = attr[FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
    constant_mem = attr[FUNC_ATTRIBUTE_CONST_SIZE_BYTES]
    return (:local=>local_mem, shared=shared_mem, constant=constant_mem)
end

"""
    registers(k::HostKernel)

Queries the register usage of a kernel.
"""
function registers(k::HostKernel)
    attr = attributes(k.fun)
    return attr[FUNC_ATTRIBUTE_NUM_REGS]
end

"""
    maxthreads(k::HostKernel)

Queries the maximum amount of threads a kernel can use in a single block.
"""
function maxthreads(k::HostKernel)
    attr = attributes(k.fun)
    return attr[FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
end


## host-side API

const cufunction_lock = ReentrantLock()

"""
    cufunction(f, tt=Tuple{}; kwargs...)

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object. For a higher-level interface, use [`@cuda`](@ref).

The following keyword arguments are supported:
- `minthreads`: the required number of threads in a thread block
- `maxthreads`: the maximum number of threads in a thread block
- `blocks_per_sm`: a minimum number of thread blocks to be scheduled on a single
  multiprocessor
- `maxregs`: the maximum number of registers to be allocated to a single thread (only
  supported on LLVM 4.0+)
- `name`: override the name that the kernel will have in the generated code
- `always_inline`: inline all function calls in the kernel
- `fastmath`: use less precise square roots and flush denormals
- `cap` and `ptx`: to override the compute capability and PTX version to compile for

The output of this function is automatically cached, i.e. you can simply call `cufunction`
in a hot path without degrading performance. New code will be generated automatically, when
when function changes, or when different types or keyword arguments are provided.
"""
function cufunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    cuda = active_state()

    Base.@lock cufunction_lock begin
        # compile the function
        cache = compiler_cache(cuda.context)
        source = methodinstance(F, tt)
        config = compiler_config(cuda.device; kwargs...)::CUDACompilerConfig
        fun = GPUCompiler.cached_compilation(cache, source, config, compile, link)

        # create a callable object that captures the function instance. we don't need to think
        # about world age here, as GPUCompiler already does and will return a different object
        key = (objectid(source), hash(fun), f)
        kernel = get(_kernel_instances, key, nothing)
        if kernel === nothing
            # create the kernel state object
            state = KernelState(create_exceptions!(fun.mod), UInt32(0))

            kernel = HostKernel{F,tt}(f, fun, state)
            _kernel_instances[key] = kernel
        end
        return kernel::HostKernel{F,tt}
    end
end

# cache of kernel instances
const _kernel_instances = Dict{Any, Any}()

function (kernel::HostKernel)(args::Vararg{Any,N}; threads::CuDim=1, blocks::CuDim=1, kwargs...) where {N}
    call(kernel, map(cudaconvert, args)...; threads, blocks, kwargs...)
end

make_seed(::HostKernel) = Random.rand(UInt32)


## device-side kernels

struct DeviceKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::CuDeviceFunction
    state::KernelState
end

@doc (@doc AbstractKernel) DeviceKernel


## device-side API

"""
    dynamic_cufunction(f, tt=Tuple{})

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object. Device-side equivalent of [`CUDA.cufunction`](@ref).

No keyword arguments are supported.
"""
@inline function dynamic_cufunction(f::F, tt::Type=Tuple{}) where {F <: Function}
    fptr = GPUCompiler.deferred_codegen(Val(F), Val(tt))
    fun = CuDeviceFunction(fptr)
    DeviceKernel{F,tt}(f, fun, kernel_state())
end

@inline (kernel::DeviceKernel)(args::Vararg{Any,N}; kwargs...) where {N} =
    call(kernel, args...; kwargs...)

# re-use the parent kernel's seed to avoid need for the RNG
make_seed(::DeviceKernel) = kernel_state().random_seed


## other

"""
    nextwarp(dev, threads)
    prevwarp(dev, threads)

Returns the next or previous nearest number of threads that is a multiple of the warp size
of a device `dev`. This is a common requirement when using intra-warp communication.
"""
function nextwarp(dev::CuDevice, threads::Integer)
    ws = warpsize(dev)
    return threads + (ws - threads % ws) % ws
end

@doc (@doc nextwarp) function prevwarp(dev::CuDevice, threads::Integer)
    ws = warpsize(dev)
    return threads - Base.rem(threads, ws)
end
