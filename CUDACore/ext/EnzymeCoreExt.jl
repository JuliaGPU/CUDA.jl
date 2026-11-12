# compatibility with EnzymeCore
module EnzymeCoreExt

using CUDACore
import CUDACore: GPUCompiler, CUDABackend

using EnzymeCore
using EnzymeCore.EnzymeRules

function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(CUDACore.context!), args...)
    return nothing
end
function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(CUDACore.is_pinned), args...)
    return nothing
end
function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(CUDACore.device_synchronize), args...)
    return nothing
end
function EnzymeCore.EnzymeRules.inactive(::typeof(CUDACore.launch_configuration), args...; kwargs...)
    return nothing
end
function EnzymeCore.EnzymeRules.inactive(::typeof(synchronize), args...; kwargs...)
    return nothing
end
# compiling a kernel has no derivative; launching the resulting kernel object is
# differentiated by the `HostKernel` rules below
function EnzymeCore.EnzymeRules.inactive(::typeof(cufunction), args...; kwargs...)
    return nothing
end

function EnzymeCore.compiler_job_from_backend(backend::CUDABackend, @nospecialize(F::Type), @nospecialize(TT::Type))
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, CUDACore.compiler_config(CUDACore.device();
                                                               fastmath=backend.fastmath))
end

function metaf(config, fn, args::Vararg{Any, N}) where N
    EnzymeCore.autodiff_deferred(EnzymeCore.set_runtime_activity(Forward, config), Const(fn), Const, args...)
    nothing
end


## rule results

# Most rules below return a primal and a shadow for each batch lane, of which the config
# says which are needed. These helpers assemble the return value of a rule from a thunk
# computing the primal and a function computing the shadow of lane `i`.

@inline function forward_return(config, primal::P, shadow::S) where {P, S}
    width = EnzymeRules.width(config)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if width == 1
            Duplicated(primal(), shadow(1))
        else
            BatchDuplicated(primal(), ntuple(shadow, Val(width)))
        end
    elseif EnzymeRules.needs_shadow(config)
        width == 1 ? shadow(1) : ntuple(shadow, Val(width))
    elseif EnzymeRules.needs_primal(config)
        primal()
    else
        nothing
    end
end

@inline function augmented_return(config, ::Type{RT}, primal::P, shadow::S,
                                  tape=nothing) where {RT, P, S}
    width = EnzymeRules.width(config)
    p = EnzymeRules.needs_primal(config) ? primal() : nothing
    s = if EnzymeRules.needs_shadow(config)
        width == 1 ? shadow(1) : ntuple(shadow, Val(width))
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       typeof(tape)}(p, s, tape)
end

# the shadow of an annotated argument in batch lane `i`
@inline shadow_lane(config, x, i) = EnzymeRules.width(config) == 1 ? x.dval : x.dval[i]


## argument conversion

# the shadow of a constant argument is a zero of the same shape
@inline converted_shadow(config, x::Const, i) = cudaconvert(EnzymeCore.make_zero(x.val))
@inline converted_shadow(config, x, i) = cudaconvert(shadow_lane(config, x, i))

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(cudaconvert)},
                                        ::Type{RT}, x::IT) where {RT, IT}
    forward_return(config, () -> ofn.val(x.val), i -> converted_shadow(config, x, i))
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(cudaconvert)}, ::Type{RT}, x::IT) where {RT, IT}
    augmented_return(config, RT, () -> ofn.val(x.val), i -> converted_shadow(config, x, i))
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(cudaconvert)}, ::Type{RT}, tape, x::IT) where {RT, IT}
    (nothing,)
end


## array construction

# The shadow of a freshly allocated array is zeroed bytewise, like Enzyme zeroes the
# shadows of CPU allocations: `zero(T)` is not defined for an arbitrary isbits struct.
@inline function zeroed_shadow(::Type{CT}, ctor, uval, primargs) where {CT}
    shadow = ctor(uval.val, primargs...)::CT
    fill!(reinterpret(UInt8, vec(shadow)), 0x00)
    return shadow
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{UndefInitializer}, args...) where {CT <: CuArray, RT}
    primargs = map(arg -> arg.val, args)
    forward_return(config, () -> ofn.val(uval.val, primargs...),
                   i -> zeroed_shadow(CT, ofn.val, uval, primargs))
end

# an array wrapping existing memory has the array wrapping the shadow memory as its shadow
function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDACore.DataRef, RT}
    primargs = map(arg -> arg.val, args)
    forward_return(config, () -> ofn.val(uval.val, primargs...; kwargs...),
                   i -> ofn.val(shadow_lane(config, uval, i), primargs...; kwargs...))
end

## kernel launches

# A kernel launch is differentiated by launching a meta-kernel instead, which differentiates
# the kernel function on the device. For reverse mode, the forward meta-kernel stores one
# tape entry per thread in a `CuArray`, which the reverse meta-kernel then consumes.
#
# The meta-kernels take the annotated arguments as they are passed to the rule; the
# host-to-device conversion of `KernelCall` recurses into the annotations, so both the
# primal and the shadow arrays are tracked by the launch's managed-memory bookkeeping.

function meta_augf(config, f, ::Val{ModifiedBetween}, tape::CuDeviceArray{TapeType},
                   args::Vararg{Any, N}) where {ModifiedBetween, TapeType, N}
    forward, _ = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        map(typeof, args)...,
    )

    @inbounds tape[thread_index()] = forward(Const(f), args...)[1]
    nothing
end

function meta_revf(config, f, ::Val{ModifiedBetween}, tape::CuDeviceArray{TapeType},
                   args::Vararg{Any, N}) where {ModifiedBetween, TapeType, N}
    _, reverse = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        map(typeof, args)...,
    )

    reverse(Const(f), args..., @inbounds tape[thread_index()])
    nothing
end

# linear index of the current thread across the whole grid, starting at 1
@inline function thread_index()
    idx = 0
    # idx *= gridDim().x
    idx += blockIdx().x-1

    idx *= gridDim().y
    idx += blockIdx().y-1

    idx *= gridDim().z
    idx += blockIdx().z-1

    idx *= blockDim().x
    idx += threadIdx().x-1

    idx *= blockDim().y
    idx += threadIdx().y-1

    idx *= blockDim().z
    idx += threadIdx().z-1
    return idx + 1
end

# kernel argument types after host-to-device conversion
@inline device_types(args...) = map(arg -> typeof(cudaconvert(arg)), args)

@inline function launch_meta(meta, launch_kwargs, compiler_kwargs, args...)
    call = CUDACore.KernelCall(meta, args...)
    kernel = CUDACore.kernel_compile(call; compiler_kwargs...)
    CUDACore.kernel_launch(kernel, call; launch_kwargs...)
    return
end

function forward_launch(config, f::F, launch_kwargs, compiler_kwargs,
                        args::Vararg{Any, N}) where {F, N}
    launch_meta(metaf, launch_kwargs, compiler_kwargs, config, f, args...)
end

# `ModifiedBetween` describes `(f, args...)`, matching the meta-kernel's differentiated call
function augmented_launch(config, f::F, ::Val{ModifiedBetween}, launch_kwargs, compiler_kwargs,
                          args::Vararg{Any, N}) where {F, ModifiedBetween, N}
    TapeType = EnzymeCore.tape_type(
        EnzymeCore.compiler_job_from_backend(CUDABackend(), typeof(Base.identity), Tuple{Float64}),
        ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        Const{F},
        Const{Nothing},
        device_types(args...)...,
    )
    threads = CuDim3(get(launch_kwargs, :threads, 1))
    blocks = CuDim3(get(launch_kwargs, :blocks, 1))
    subtape = CuArray{TapeType}(undef, blocks.x*blocks.y*blocks.z*threads.x*threads.y*threads.z)

    launch_meta(meta_augf, launch_kwargs, compiler_kwargs,
                config, f, Val(ModifiedBetween), subtape, args...)
    return subtape
end

function reverse_launch(config, f::F, ::Val{ModifiedBetween}, subtape, launch_kwargs, compiler_kwargs,
                        args::Vararg{Any, N}) where {F, ModifiedBetween, N}
    launch_meta(meta_revf, launch_kwargs, compiler_kwargs,
                config, f, Val(ModifiedBetween), subtape, args...)
end

# the kernel object returned by a launch is inactive; hand it back as primal and/or shadow
@inline kernel_result(compile, config, ::Type{RT}) where {RT} =
    forward_return(config, compile, _ -> compile())

@inline kernel_augmented_result(compile, config, ::Type{RT}, tape) where {RT} =
    augmented_return(config, RT, compile, _ -> compile(), tape)


### `@cuda`

# `@cuda` expands to `kernel_pipeline`, which receives the host function and the
# un-converted host arguments as individual positional arguments. Hooking it keeps
# Enzyme out of argument conversion, compilation, and the managed-memory bookkeeping
# of the launch, none of which is differentiable.

const KernelPipeline = typeof(CUDACore.kernel_pipeline)

# `overwritten(config)` covers `(kernel_pipeline, backend, f, Val(launch), args...)`,
# whereas the meta-kernels differentiate `f(args...)`
@inline function pipeline_overwritten(config)
    ow = EnzymeRules.overwritten(config)
    return (ow[3], ow[5:end]...)
end

@inline function primal_kernel(backend, f, compiler_kwargs, args...)
    tt = Tuple{map(arg -> Core.Typeof(cudaconvert(arg.val)), args)...}
    return CUDACore.kernel_compile(backend, f, tt; compiler_kwargs...)
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{KernelPipeline}, ::Type{RT},
                                        backend::Const{CUDACore.LLVMBackend},
                                        f::EnzymeCore.Annotation{F}, ::Const{Val{launch}},
                                        args::Vararg{EnzymeCore.Annotation, N};
                                        launch_kwargs::NamedTuple=(;),
                                        compiler_kwargs...) where {RT, F, launch, N}
    if launch
        forward_launch(config, f.val, launch_kwargs, compiler_kwargs, args...)
    end
    return kernel_result(config, RT) do
        primal_kernel(backend.val, f.val, compiler_kwargs, args...)
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{KernelPipeline}, ::Type{RT},
                                                 backend::Const{CUDACore.LLVMBackend},
                                                 f::EnzymeCore.Annotation{F}, ::Const{Val{launch}},
                                                 args::Vararg{EnzymeCore.Annotation, N};
                                                 launch_kwargs::NamedTuple=(;),
                                                 compiler_kwargs...) where {RT, F, launch, N}
    tape = if launch
        augmented_launch(config, f.val, Val(pipeline_overwritten(config)),
                         launch_kwargs, compiler_kwargs, args...)
    else
        nothing
    end
    return kernel_augmented_result(config, RT, tape) do
        primal_kernel(backend.val, f.val, compiler_kwargs, args...)
    end
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{KernelPipeline}, ::Type{RT}, tape,
                                        backend::Const{CUDACore.LLVMBackend},
                                        f::EnzymeCore.Annotation{F}, ::Const{Val{launch}},
                                        args::Vararg{EnzymeCore.Annotation, N};
                                        launch_kwargs::NamedTuple=(;),
                                        compiler_kwargs...) where {RT, F, launch, N}
    if launch
        reverse_launch(config, f.val, Val(pipeline_overwritten(config)), tape,
                       launch_kwargs, compiler_kwargs, args...)
    end
    return ntuple(_ -> nothing, Val(N + 3))
end


### compiled kernel objects

# a kernel object obtained from `@cuda launch=false` or `cufunction` is launched by
# calling it with host arguments, converting them along the way

function EnzymeCore.EnzymeRules.forward(config, ofn::EnzymeCore.Annotation{CUDACore.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}},
                                        args::Vararg{EnzymeCore.Annotation, N};
                                        kwargs...) where {F, TT, N}
    forward_launch(config, ofn.val.f, (; kwargs...), (;), args...)
    return nothing
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::EnzymeCore.Annotation{CUDACore.HostKernel{F,TT}},
                                                 ::Type{Const{Nothing}},
                                                 args::Vararg{EnzymeCore.Annotation, N};
                                                 kwargs...) where {F, TT, N}
    # `overwritten(config)` covers `(kernel, args...)`, matching `(f, args...)`
    subtape = augmented_launch(config, ofn.val.f, Val(EnzymeRules.overwritten(config)),
                               (; kwargs...), (;), args...)
    return AugmentedReturn{Nothing,Nothing,CuArray}(nothing, nothing, subtape)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::EnzymeCore.Annotation{CUDACore.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, subtape,
                                        args::Vararg{EnzymeCore.Annotation, N};
                                        kwargs...) where {F, TT, N}
    reverse_launch(config, ofn.val.f, Val(EnzymeRules.overwritten(config)), subtape,
                   (; kwargs...), (;), args...)
    return ntuple(_ -> nothing, Val(N))
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{Type{CT}}, ::Type{RT}, uval::EnzymeCore.Annotation{UndefInitializer}, args...) where {CT <: CuArray, RT}
    primargs = map(arg -> arg.val, args)
    augmented_return(config, RT, () -> ofn.val(uval.val, primargs...)::CT,
                     i -> zeroed_shadow(CT, ofn.val, uval, primargs))
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape, A::EnzymeCore.Annotation{UndefInitializer}, args::Vararg{EnzymeCore.Annotation, N}) where {CT <: CuArray, RT, N}
    ntuple(Val(N+1)) do i
          Base.@_inline_meta
          nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{Type{CT}}, ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDACore.DataRef, RT}
    primargs = map(arg -> arg.val, args)
    augmented_return(config, RT, () -> ofn.val(uval.val, primargs...; kwargs...),
                     i -> ofn.val(shadow_lane(config, uval, i), primargs...; kwargs...))
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape, A::EnzymeCore.Annotation{DR}, args::Vararg{EnzymeCore.Annotation, N}; kwargs...) where {CT <: CuArray, DR <: CUDACore.DataRef, RT, N}
    ntuple(Val(N+1)) do i
          Base.@_inline_meta
          nothing
    end
end

### reference boxes

# `CuRefValue{T}()` allocates its buffer through `pool_alloc`, which also updates the
# global allocation statistics using atomic operations on `Float64` fields that Enzyme
# cannot differentiate. Allocating a box has no derivative of its own, but the box may later
# hold an active value, so we cannot mark the constructor inactive: give it a zero-initialized
# shadow box instead, like the `CuArray(undef, ...)` constructor rules above.

function zero_curef!(ref::CUDACore.CuRefValue)
    GC.@preserve ref CUDACore.memset(convert(CuPtr{UInt8}, ref.buf), 0x00, sizeof(ref.buf))
    return ref
end

zeroed_curef(::Type{CT}) where {CT <: CUDACore.CuRefValue} = zero_curef!(CT())

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}}, ::Type{RT}) where {CT <: CUDACore.CuRefValue, RT}
    forward_return(config, () -> ofn.val(), _ -> zeroed_curef(CT))
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{Type{CT}}, ::Type{RT}) where {CT <: CUDACore.CuRefValue, RT}
    augmented_return(config, RT, () -> ofn.val(), _ -> zeroed_curef(CT))
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape) where {CT <: CUDACore.CuRefValue, RT}
    return ()
end

# Enzyme loses the shadow when taking the device address of a buffer (which also takes
# stream ownership of it), so return the address of the shadow buffer explicitly. Copies
# from and to that address are then differentiated by the `unsafe_copyto!` rules in
# Enzyme.jl, e.g. when reading or writing a `CuRefValue`.

managed_shadow_ptr(config, ::Type{P}, managed::Const, i) where {P} = convert(P, managed.val)
managed_shadow_ptr(config, ::Type{P}, managed, i) where {P} =
    convert(P, shadow_lane(config, managed, i))

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(Base.convert)}, ::Type{RT},
                                        ::Const{Type{P}}, managed::EnzymeCore.Annotation{<:CUDACore.Managed}) where {RT, P <: CuPtr}
    forward_return(config, () -> convert(P, managed.val),
                   i -> managed_shadow_ptr(config, P, managed, i))
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(Base.convert)}, ::Type{RT},
                                                 ::Const{Type{P}}, managed::EnzymeCore.Annotation{<:CUDACore.Managed}) where {RT, P <: CuPtr}
    augmented_return(config, RT, () -> convert(P, managed.val),
                     i -> managed_shadow_ptr(config, P, managed, i))
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(Base.convert)}, ::Type{RT}, tape,
                                        ::Const{Type{P}}, managed::EnzymeCore.Annotation{<:CUDACore.Managed}) where {RT, P <: CuPtr}
    return (nothing, nothing)
end

function EnzymeCore.EnzymeRules.noalias(::Type{CT}, ::UndefInitializer, args...) where {CT <: CuArray}
    return nothing
end


end # module

