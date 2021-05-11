# Native execution support

export @cuda, cudaconvert, cufunction, dynamic_cufunction, nextwarp, prevwarp


## high-level @cuda interface

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
    kwargs = ex[1:end-1]

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @cuda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs,
                     [:dynamic, :launch],
                     [:minthreads, :maxthreads, :blocks_per_sm, :maxregs, :name],
                     [:cooperative, :blocks, :threads, :shmem, :stream])
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    dynamic = false
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key == :dynamic
            isa(val, Bool) || throw(ArgumentError("`dynamic` keyword argument to @cuda should be a constant value"))
            dynamic = val::Bool
        elseif key == :launch
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
                local $kernel_args = ($(var_exprs...),)
                local $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                local $kernel = $dynamic_cufunction($f, $kernel_tt)
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
                    local $kernel_f = $cudaconvert($f_var)
                    local $kernel_args = map($cudaconvert, ($(var_exprs...),))
                    local $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                    local $kernel = $cufunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                    if $launch
                        $kernel($(var_exprs...); $(call_kwargs...))
                    end
                    $kernel
                end
             end)
    end
    return esc(code)
end


## host to device value conversion

struct Adaptor end

# convert CUDA host pointers to device pointers
# TODO: use ordinary ptr?
Adapt.adapt_storage(to::Adaptor, p::CuPtr{T}) where {T} = reinterpret(LLVMPtr{T,AS.Generic}, p)

# Base.RefValue isn't GPU compatible, so provide a compatible alternative
struct CuRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::CuRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = CuRefValue(adapt(to, r[]))

Adapt.adapt_storage(::Adaptor, xs::CuArray{T,N}) where {T,N} =
  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)

# we materialize ReshapedArray/ReinterpretArray/SubArray/... directly as a device array
Adapt.adapt_structure(::Adaptor, xs::DenseCuArray{T,N}) where {T,N} =
  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)

"""
    cudaconvert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `CUDA.Adaptor` type.
"""
cudaconvert(arg) = adapt(Adaptor(), arg)


## abstract kernel functionality

abstract type AbstractKernel{F,TT} end

# FIXME: there doesn't seem to be a way to access the documentation for the call-syntax,
#        so attach it to the type -- https://github.com/JuliaDocs/Documenter.jl/issues/558

"""
    (::HostKernel)(args...; kwargs...)
    (::DeviceKernel)(args...; kwargs...)

Low-level interface to call a compiled kernel, passing GPU-compatible arguments in `args`.
For a higher-level interface, use [`@cuda`](@ref).

The following keyword arguments are supported:
- `threads` (defaults to 1)
- `blocks` (defaults to 1)
- `shmem` (defaults to 0)
- `stream` (defaults to the default stream)
"""
AbstractKernel

@generated function call(kernel::AbstractKernel{F,TT}, args...; call_kwargs...) where {F,TT}
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

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        Base.@_inline_meta

        cudacall(kernel.fun, $call_tt, $(call_args...); call_kwargs...)
    end
end


## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    ctx::CuContext
    mod::CuModule
    fun::CuFunction
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

The output of this function is automatically cached, i.e. you can simply call `cufunction`
in a hot path without degrading performance. New code will be generated automatically, when
when function changes, or when different types or keyword arguments are provided.
"""
@timeit_debug to function cufunction(f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where
                                    {F<:Core.Function, TT<:Type}
    dev = device()
    cache = cufunction_cache[dev]
    source = FunctionSpec(f, tt, true, name)
    target = CUDACompilerTarget(dev; kwargs...)
    params = CUDACompilerParams()
    job = CompilerJob(target, source, params)
    return GPUCompiler.cached_compilation(cache, job,
                                          cufunction_compile,
                                          cufunction_link)::HostKernel{F,tt}
end

const cufunction_cache = PerDevice{Dict{UInt, Any}}((dev)->Dict{UInt, Any}())

# compile to PTX
@timeit_debug to "compile" function cufunction_compile(@nospecialize(job::CompilerJob))
    # compile
    method_instance, world = @timeit_debug to "emit_julia" GPUCompiler.emit_julia(job)
    ir, kernel = @timeit_debug to "emit_llvm" GPUCompiler.emit_llvm(job, method_instance, world)
    code = @timeit_debug to "emit_asm" GPUCompiler.emit_asm(job, ir, kernel; format=LLVM.API.LLVMAssemblyFile)

    # check if we'll need the device runtime
    undefined_fs = filter(collect(functions(ir))) do f
        isdeclaration(f) && !LLVM.isintrinsic(f)
    end
    intrinsic_fns = ["vprintf", "malloc", "free", "__assertfail",
                    "__nvvm_reflect" #= TODO: should have been optimized away =#]
    needs_cudadevrt = !isempty(setdiff(LLVM.name.(undefined_fs), intrinsic_fns))

    # find externally-initialized global variables; we'll access those using CUDA APIs.
    external_gvars = filter(isextinit, collect(globals(ir))) .|> LLVM.name

    return (code, entry=LLVM.name(kernel), needs_cudadevrt, external_gvars)
end

# link to device code
@timeit_debug to "link" function cufunction_link(@nospecialize(job::CompilerJob), compiled)
    ctx = context()

    # settings to JIT based on Julia's debug setting
    jit_options = Dict{CUjit_option,Any}()
    if Base.JLOptions().debug_level == 1
        jit_options[JIT_GENERATE_LINE_INFO] = true
    elseif Base.JLOptions().debug_level >= 2
        jit_options[JIT_GENERATE_DEBUG_INFO] = true
    end

    # link the CUDA device library
    # linking the device runtime library requires use of the CUDA linker,
    # which in turn switches compilation to device relocatable code (-rdc) mode.
    #
    # even if not doing any actual calls that need -rdc (i.e., calls to the runtime
    # library), this significantly hurts performance, so don't do it unconditionally
    intrinsic_fns = ["vprintf", "malloc", "free", "__assertfail",
                    "__nvvm_reflect" #= TODO: should have been optimized away =#]
    image = if compiled.needs_cudadevrt
        @timeit_debug to "cudadevrt" begin
        linker = CuLink(jit_options)
        add_file!(linker, libcudadevrt(), JIT_INPUT_LIBRARY)
        add_data!(linker, compiled.entry, compiled.code)
        complete(linker)
        end
    else
        compiled.code
    end

    # JIT into an executable kernel object
    mod = @timeit_debug to "CuModule" CuModule(image, jit_options)
    fun = CuFunction(mod, compiled.entry)

    # initialize and register the exception flag, if any
    if "exception_flag" in compiled.external_gvars
        create_exceptions!(mod)
        filter!(!isequal("exception_flag"), compiled.external_gvars)
    end

    # initialize random seeds, if used
    if "global_random_seed" in compiled.external_gvars
        random_state = missing
        initialize_random_seeds!(mod)
        filter!(!isequal("global_random_seed"), compiled.external_gvars)
    end

    return HostKernel{typeof(job.source.f),job.source.tt}(job.source.f, ctx, mod, fun)
end

function (kernel::HostKernel)(args...; threads::CuDim=1, blocks::CuDim=1, kwargs...)
    call(kernel, map(cudaconvert, args)...; threads, blocks, kwargs...)
end


## device-side kernels

struct DeviceKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::CuDeviceFunction
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
    fptr = GPUCompiler.deferred_codegen(Val(f), Val(tt))
    fun = CuDeviceFunction(fptr)
    DeviceKernel{F,tt}(f, fun)
end

(kernel::DeviceKernel)(args...; kwargs...) = call(kernel, args...; kwargs...)


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
