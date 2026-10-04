# compatibility with EnzymeCore
module EnzymeCoreExt

using CUDACore
import CUDACore: GPUCompiler, CUDABackend

if isdefined(Base, :get_extension)
    using EnzymeCore
    using EnzymeCore.EnzymeRules
else
    using ..EnzymeCore
    using ..EnzymeCore.EnzymeRules
end

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

function EnzymeCore.compiler_job_from_backend(backend::CUDABackend, @nospecialize(F::Type), @nospecialize(TT::Type))
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, CUDACore.compiler_config(CUDACore.device();
                                                               fastmath=backend.fastmath))
end

function metaf(config, fn, args::Vararg{Any, N}) where N
    EnzymeCore.autodiff_deferred(EnzymeCore.set_runtime_activity(Forward, config), Const(fn), Const, args...)
    nothing
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(cufunction)},
                                        ::Type{<:Const}, f::Const{F},
                                        tt::Const{TT}; kwargs...) where {F,TT}
    res = ofn.val(f.val, tt.val; kwargs...)
    return res
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(cufunction)},
                                        ::Type{<:Duplicated}, f::Const{F},
                                        tt::Const{TT}; kwargs...) where {F,TT}
    res = ofn.val(f.val, tt.val; kwargs...)
    return Duplicated(res, res)
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(cufunction)},
                                        ::Type{BatchDuplicated{T,N}}, f::Const{F},
                                        tt::Const{TT}; kwargs...) where {F,TT,T,N}
    res = ofn.val(f.val, tt.val; kwargs...)
    return BatchDuplicated(res, ntuple(Val(N)) do _
        res
    end)
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(cudaconvert)},
                                        ::Type{RT}, x::IT) where {RT, IT}

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            Duplicated(ofn.val(x.val), ofn.val(x.dval))
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                ofn.val(x.dval[i])::eltype(RT)
            end
            BatchDuplicated(ofn.val(x.val), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            ofn.val(x.dval)::EnzymeCore.shadow_type(config, RT)
        else
            (ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                ofn.val(x.dval[i])::eltype(RT)
            end)::EnzymeCore.shadow_type(config, RT)
        end
    elseif EnzymeRules.needs_primal(config)
        ofn.val(x.val)::eltype(RT)
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(cudaconvert)}, ::Type{RT}, x::IT) where {RT, IT}
    primal = if EnzymeRules.needs_primal(config)
        ofn.val(x.val)
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1 && !(IT <: Const)
            ofn.val(x.dval)
        elseif EnzymeRules.width(config) == 1 && IT <: Const
            ofn.val(EnzymeCore.make_zero(x.val))
        elseif !(IT <: Const)
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              ofn.val(x.dval[i])
          end
        else
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              ofn.val(EnzymeCore.make_zero(x.val[i]))
          end
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(cudaconvert)}, ::Type{RT}, tape, x::IT) where {RT, IT}
    (nothing,)
end


function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{UndefInitializer}, args...) where {CT <: CuArray, RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = ofn.val(uval.val, primargs...)::CT
            fill!(shadow, zero(eltype(shadow)))
            Duplicated(ofn.val(uval.val, primargs...), shadow)
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                shadow = ofn.val(uval.val, primargs...)::CT
                fill!(shadow, zero(eltype(shadow)))
                shadow::CT
            end
            BatchDuplicated(ofn.val(uval.val, primargs...), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = ofn.val(uval.val, primargs...)::CT
            fill!(shadow, zero(eltype(shadow)))
	    shadow::shadow_type(config, RT)
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                shadow = ofn.val(uval.val, primargs...)::CT
                fill!(shadow, zero(eltype(shadow)))
                shadow::CT
            end
	    tup::shadow_type(config, RT)
        end
    elseif EnzymeRules.needs_primal(config)
        ofn.val(uval.val, primargs...)
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDACore.DataRef, RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = ofn.val(uval.val, primargs...; kwargs...)
            Duplicated(ofn.val(uval.val, primargs...; kwargs...), shadow)
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                ofn.val(uval.val, primargs...; kwargs...)
            end
            BatchDuplicated(ofn.val(uval.val, primargs...; kwargs...), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = ofn.val(uval.val, primargs...; kwargs...)
            shadow
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                ofn.val(uval.val, primargs...; kwargs...)
            end
            tup
        end
    elseif EnzymeRules.needs_primal(config)
        ofn.val(uval.val, primargs...; kwargs...)
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(synchronize)},
                                        ::Type{RT}, args::Vararg{EnzymeCore.Annotation, N}; kwargs...) where {RT, N}
    pargs = ntuple(Val(N)) do i
        Base.@_inline_meta
        args[i].val
    end
    res = ofn.val(pargs...; kwargs...)

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            Duplicated(res, res)
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                res
            end
            BatchDuplicated(ofn.val(uval.val, primargs...; kwargs...), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            res
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                res
            end
        end
    elseif EnzymeRules.needs_primal(config)
        res
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(cufunction)},
                                            ::Type{RT}, f::Const{F},
                                            tt::Const{TT}; kwargs...) where {F,CT, RT<:EnzymeCore.Annotation{CT}, TT}
    res = ofn.val(f.val, tt.val; kwargs...)

    primal = if EnzymeRules.needs_primal(config)
        res
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            res
        else
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              res
          end
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::EnzymeCore.Const{typeof(cufunction)},::Type{RT}, subtape, f, tt; kwargs...) where RT
    return (nothing, nothing)
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
@inline function kernel_result(compile, config, ::Type{RT}) where {RT}
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        kernel = compile()
        if EnzymeRules.width(config) == 1
            Duplicated(kernel, kernel)
        else
            BatchDuplicated(kernel, ntuple(_ -> kernel, Val(EnzymeRules.width(config))))
        end
    elseif EnzymeRules.needs_shadow(config)
        kernel = compile()
        if EnzymeRules.width(config) == 1
            kernel
        else
            ntuple(_ -> kernel, Val(EnzymeRules.width(config)))
        end
    elseif EnzymeRules.needs_primal(config)
        compile()
    else
        nothing
    end
end

@inline function kernel_augmented_result(compile, config, ::Type{RT}, tape) where {RT}
    primal = EnzymeRules.needs_primal(config) ? compile() : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        kernel = compile()
        EnzymeRules.width(config) == 1 ? kernel : ntuple(_ -> kernel, Val(EnzymeRules.width(config)))
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       typeof(tape)}(primal, shadow, tape)
end


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
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end

    primal = if EnzymeRules.needs_primal(config)
        ofn.val(uval.val, primargs...)::CT
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            subshadow = ofn.val(uval.val, primargs...)::CT
            fill!(subshadow, zero(eltype(subshadow)))
            subshadow
        else
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              subshadow = ofn.val(uval.val, primargs...)::CT
              fill!(subshadow, zero(eltype(subshadow)))
              subshadow
          end
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape, A::EnzymeCore.Annotation{UndefInitializer}, args::Vararg{EnzymeCore.Annotation, N}) where {CT <: CuArray, RT, N}
    ntuple(Val(N+1)) do i
          Base.@_inline_meta
          nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{Type{CT}}, ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDACore.DataRef, RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end

    primal = if EnzymeRules.needs_primal(config)
        ofn.val(uval.val, primargs...; kwargs...)
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            ofn.val(uval.dval, primargs...; kwargs...)
        else
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              ofn.val(uval.dval[i], primargs...; kwargs...)
          end
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
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

function zeroed_curef(::Type{CT}) where {CT <: CUDACore.CuRefValue}
    return zero_curef!(CT())
end

function curef_shadow(config, ::Type{CT}) where {CT <: CUDACore.CuRefValue}
    if EnzymeRules.width(config) == 1
        zeroed_curef(CT)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zeroed_curef(CT)
        end
    end
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}}, ::Type{RT}) where {CT <: CUDACore.CuRefValue, RT}
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            Duplicated(ofn.val(), curef_shadow(config, CT))
        else
            BatchDuplicated(ofn.val(), curef_shadow(config, CT))
        end
    elseif EnzymeRules.needs_shadow(config)
        curef_shadow(config, CT)
    elseif EnzymeRules.needs_primal(config)
        ofn.val()
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{Type{CT}}, ::Type{RT}) where {CT <: CUDACore.CuRefValue, RT}
    primal = EnzymeRules.needs_primal(config) ? ofn.val() : nothing
    shadow = EnzymeRules.needs_shadow(config) ? curef_shadow(config, CT) : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape) where {CT <: CUDACore.CuRefValue, RT}
    return ()
end

# Reading and writing a box copies between host and device memory with `cuMemcpy`, which
# Enzyme does not see through, so the derivative is propagated explicitly here: the shadow
# box holds the derivative of the stored value, a write sets it and a read passes it on.

curef_shadow_at(dval, ::Val{1}, i) = dval
curef_shadow_at(dval, ::Val, i) = dval[i]

# read the derivative from (the `i`-th) shadow box, and clear it
curef_take!(ref::Const, x) = zero(x)
curef_take!(ref::Const, x, i) = zero(x)
curef_take!(ref::EnzymeCore.Annotation, x) = curef_take!(ref.dval)
curef_take!(ref::EnzymeCore.Annotation, x, i) = curef_take!(ref.dval[i])
function curef_take!(dref::CUDACore.CuRefValue{T}) where {T}
    d = dref[]::T
    zero_curef!(dref)
    return d
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(Base.setindex!)}, ::Type{RT},
                                        ref::EnzymeCore.Annotation{CT}, x::EnzymeCore.Annotation{T}) where {T, CT <: CUDACore.CuRefValue{T}, RT}
    ref.val[] = x.val
    if !(ref isa Const)
        W = Val(EnzymeRules.width(config))
        for i in 1:EnzymeRules.width(config)
            dref = curef_shadow_at(ref.dval, W, i)
            if x isa Const
                zero_curef!(dref)
            else
                dref[] = curef_shadow_at(x.dval, W, i)
            end
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            Duplicated(ref.val, ref.dval)
        else
            BatchDuplicated(ref.val, ref.dval)
        end
    elseif EnzymeRules.needs_shadow(config)
        ref.dval
    elseif EnzymeRules.needs_primal(config)
        ref.val
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(Base.setindex!)}, ::Type{RT},
                                                 ref::EnzymeCore.Annotation{CT}, x::EnzymeCore.Annotation{T}) where {T, CT <: CUDACore.CuRefValue{T}, RT}
    ref.val[] = x.val
    primal = EnzymeRules.needs_primal(config) ? ref.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ref.dval : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(Base.setindex!)}, ::Type{RT}, tape,
                                        ref::EnzymeCore.Annotation{CT}, x::EnzymeCore.Annotation{T}) where {T, CT <: CUDACore.CuRefValue{T}, RT}
    # the write overwrote the box, so its derivative moves to `x` and the shadow box is cleared
    dx = if EnzymeRules.width(config) == 1
        curef_take!(ref, x.val)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            curef_take!(ref, x.val, i)
        end
    end
    return (nothing, x isa Active ? dx : nothing)
end

curef_shadow_value(ref::Const{CT}, i...) where {T, CT <: CUDACore.CuRefValue{T}} = zero(T)
curef_shadow_value(ref::EnzymeCore.Annotation{CT}) where {T, CT <: CUDACore.CuRefValue{T}} = ref.dval[]::T
curef_shadow_value(ref::EnzymeCore.Annotation{CT}, i) where {T, CT <: CUDACore.CuRefValue{T}} = ref.dval[i][]::T

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(Base.getindex)}, ::Type{RT},
                                        ref::EnzymeCore.Annotation{CT}) where {T, CT <: CUDACore.CuRefValue{T}, RT}
    shadow = if !EnzymeRules.needs_shadow(config)
        nothing
    elseif EnzymeRules.width(config) == 1
        curef_shadow_value(ref)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            curef_shadow_value(ref, i)
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            Duplicated(ref.val[], shadow)
        else
            BatchDuplicated(ref.val[], shadow)
        end
    elseif EnzymeRules.needs_shadow(config)
        shadow
    elseif EnzymeRules.needs_primal(config)
        ref.val[]
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(Base.getindex)}, ::Type{RT},
                                                 ref::EnzymeCore.Annotation{CT}) where {T, CT <: CUDACore.CuRefValue{T}, RT <: Union{Const, Active}}
    primal = EnzymeRules.needs_primal(config) ? ref.val[] : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), Nothing, Nothing}(primal, nothing, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(Base.getindex)}, ::Type{<:Const}, tape,
                                        ref::EnzymeCore.Annotation{CT}) where {T, CT <: CUDACore.CuRefValue{T}}
    return (nothing,)
end

# in batch mode, `dret` is a tuple with one `Active` per batch element
curef_dret(dret::Active, i) = dret.val
curef_dret(dret::Tuple, i) = dret[i].val

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(Base.getindex)}, dret::Union{Active, Tuple{Vararg{Active}}}, tape,
                                        ref::EnzymeCore.Annotation{CT}) where {T, CT <: CUDACore.CuRefValue{T}}
    if !(ref isa Const)
        W = Val(EnzymeRules.width(config))
        for i in 1:EnzymeRules.width(config)
            dref = curef_shadow_at(ref.dval, W, i)
            dref[] += curef_dret(dret, i)
        end
    end
    return (nothing,)
end

function EnzymeCore.EnzymeRules.noalias(::Type{CT}, ::UndefInitializer, args...) where {CT <: CuArray}
    return nothing
end


end # module

