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
using GPUArrays

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

function EnzymeCore.compiler_job_from_backend(::CUDABackend, @nospecialize(F::Type), @nospecialize(TT::Type))
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, CUDACore.compiler_config(CUDACore.device()))
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

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(Base.fill!)}, ::Type{RT}, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x) where {RT, T <: CUDACore.MemsetCompatTypes}
    if A isa Const || A isa Duplicated || A isa BatchDuplicated
        ofn.val(A.val, x.val)
    end

    if A isa Duplicated || A isa DuplicatedNoNeed
        ofn.val(A.dval, x isa Const ? zero(T) : x.dval)
    elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.batch_width(A))) do i
            Base.@_inline_meta
            ofn.val(A.dval[i], x isa Const ? zero(T) : x.dval[i])
            nothing
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        A
    elseif EnzymeRules.needs_shadow(config)
        A.dval
    elseif EnzymeRules.needs_primal(config)
        A.val
    else
        nothing
    end
end


function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(Base.fill!)}, ::Type{RT}, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x) where {RT, T <: CUDACore.MemsetCompatTypes}
    if A isa Const || A isa Duplicated || A isa BatchDuplicated
        ofn.val(A.val, x.val)
    end

    if !(T <: AbstractFloat)
      if A isa Duplicated || A isa DuplicatedNoNeed
          ofn.val(A.dval, zero(T))
      elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
          ntuple(Val(EnzymeRules.batch_width(A))) do i
              Base.@_inline_meta
              ofn.val(A.dval[i], zero(T))
              nothing
          end
      end
    end

    primal = if EnzymeRules.needs_primal(config)
        A.val
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        A.dval
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(Base.fill!)}, ::Type{RT}, tape, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x::EnzymeCore.Annotation{T2}) where {RT, T <: CUDACore.MemsetCompatTypes, T2}
    dx = if x isa Active 
        if A isa Duplicated || A isa DuplicatedNoNeed
            T2(sum(A.dval))
        elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
            ntuple(Val(EnzymeRules.batch_width(A))) do i
                Base.@_inline_meta
                T2(sum(A.dval[i]))
            end
        end
    else
        nothing
    end

    # re-zero shadow
    if (T <: AbstractFloat)
      if A isa Duplicated || A isa DuplicatedNoNeed
          ofn.val(A.dval, zero(T))
      elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
          ntuple(Val(EnzymeRules.batch_width(A))) do i
              Base.@_inline_meta
              ofn.val(A.dval[i], zero(T))
              nothing
          end
      end
    end

    return (nothing, dx)
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

function EnzymeCore.EnzymeRules.noalias(::Type{CT}, ::UndefInitializer, args...) where {CT <: CuArray}
    return nothing
end

# `make_zero`/`make_zero!` are defined element-wise, which would require scalar
# indexing on a `CuArray`, so zero the array in bulk instead. That is only
# equivalent to the element-wise definition when every part of the element type is
# a float, which covers `Float32`, `ComplexF64`, `SVector{3,Float64}` etc.
# For other element types the element-wise definition is run on a host copy instead.
@inline float_only(::Type{FT}) where {FT <: AbstractFloat} = true
@inline float_only(::Type{Complex{FT}}) where {FT <: AbstractFloat} = true
@inline float_only(::Type{FT}) where {FT} =
    isbitstype(FT) && fieldcount(FT) > 0 && all(float_only, fieldtypes(FT))
@inline bulk_zeroable(::Type{FT}) where {FT} =
    float_only(FT) && hasmethod(Base.zero, Tuple{Type{FT}})

# An element type that is not `bulk_zeroable` can still have float content that has to be
# zeroed, e.g. any struct mixing floats with flags or indices
@inline function make_zero_via_host(prev::CT,
                                    ::Val{copy_if_inactive}) where {copy_if_inactive, CT <: DenseCuArray}
    zeroed = map(Array(prev)) do x
        EnzymeCore.make_zero(Core.Typeof(x), IdDict(), x, Val(copy_if_inactive))
    end
    newa = similar(prev)
    copyto!(newa, zeroed)
    return newa::CT
end

@inline function EnzymeCore.make_zero(
    x::DenseCuArray{FT},
) where {FT}
    if !bulk_zeroable(FT)
        return make_zero_via_host(x, Val(false))
    end
    return Base.zero(x)
end

@inline function EnzymeCore.make_zero(
    ::Type{CT},
    seen::IdDict,
    prev::CT,
    ::Val{copy_if_inactive} = Val(false),
)::CT where {copy_if_inactive, FT, CT <: DenseCuArray{FT}}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = bulk_zeroable(FT) ? Base.zero(prev) : make_zero_via_host(prev, Val(copy_if_inactive))
    seen[prev] = newa
    return newa
end

@inline function EnzymeCore.make_zero!(
    prev::DenseCuArray{FT},
    seen::ST,
)::Nothing where {FT,ST}
    if !isnothing(seen)
        if prev in seen
            return nothing
        end
        push!(seen, prev)
    end
    if bulk_zeroable(FT)
        fill!(prev, zero(FT))
    else
        copyto!(prev, map(EnzymeCore.make_zero, Array(prev)))
    end
    return nothing
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(GPUArrays.mapreducedim!)},
                                        ::Type{RT},
                                        f::EnzymeCore.Const{typeof(Base.identity)},
                                        op::EnzymeCore.Const{typeof(Base.add_sum)},
                                        R::EnzymeCore.Annotation{<:AnyCuArray{T}}, A; init) where {RT, T}
    if R isa Const || R isa Duplicated || R isa BatchDuplicated
        ofn.val(f.val, op.val, R.val, A.val; init)
    end

    if A isa Duplicated || A isa DuplicatedNoNeed
        if A isa Const
            Base.fill!(R.dval, zero(T))
        else
            ofn.val(f.val, op.val, R.dval, A.dval)
        end
    elseif R isa BatchDuplicated || R isa BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.batch_width(R))) do i
            Base.@_inline_meta
            if A isa Const
                Base.fill!(R.dval[i], zero(T))
            else
                ofn.val(f.val, op.val, R.dval[i], A.dval[i])
            end
            nothing
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        R
    elseif EnzymeRules.needs_shadow(config)
        R.dval
    elseif EnzymeRules.needs_primal(config)
        R.val
    else
        nothing
    end
end


function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(GPUArrays.mapreducedim!)},
                                        ::Type{RT},
                                        f::EnzymeCore.Const{typeof(Base.identity)},
                                        op::EnzymeCore.Const{typeof(Base.add_sum)},
                                        R::EnzymeCore.Annotation{<:AnyCuArray{T}}, A; init) where {RT, T}
    if A isa Const || A isa Duplicated || A isa BatchDuplicated
        ofn.val(f.val, op.val, R.val, A.val)
    end

    primal = if EnzymeRules.needs_primal(config)
        R.val
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        R.dval
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(GPUArrays.mapreducedim!)},
                                        ::Type{RT},
                                        tape,
                                        f::EnzymeCore.Const{typeof(Base.identity)},
                                        op::EnzymeCore.Const{typeof(Base.add_sum)},
                                        R::EnzymeCore.Annotation{<:AnyCuArray{T}}, A; init) where {RT, T}

    if !(A isa Const) && !(R isa Const)
        if A isa Duplicated || A isa DuplicatedNoNeed
            A.dval .+= R.dval
            Base.fill!(R.dval, zero(T))
        elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
            ntuple(Val(EnzymeRules.batch_width(A))) do i
                Base.@_inline_meta
                A.dval[i] .+= R.dval[i]
                Base.fill!(R.dval[i], zero(T))
                nothing
            end
        end
    end

    return (nothing, nothing, nothing, nothing)
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(GPUArrays._mapreduce)},
                                        ::Type{RT},
                                        f::EnzymeCore.Const{typeof(Base.identity)},
                                        op::EnzymeCore.Const{typeof(Base.add_sum)},
                                        A::EnzymeCore.Annotation{<:AnyCuArray{T}}; dims::D, init) where {RT, T, D}

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = ofn.val(f.val, op.val, A.dval; dims, init)
            Duplicated(ofn.val(f.val, op.val, A.val; dims, init), shadow)
        else
            tup = ntuple(Val(EnzymeRules.batch_width(RT))) do i
                Base.@_inline_meta
                ofn.val(f.val, op.val, A.dval[i]; dims, init)
             end
            BatchDuplicated(ofn.val(f.val, op.val, A.val; dims, init), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            ofn.val(f.val, op.val, A.dval; dims, init)
        else
            ntuple(Val(EnzymeRules.batch_width(RT))) do i
                Base.@_inline_meta
                ofn.val(f.val, op.val, A.dval[i]; dims, init)
            end
        end
    elseif EnzymeRules.needs_primal(config)
        ofn.val(f.val, op.val, A.val; dims, init)
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(GPUArrays._mapreduce)},
                                                ::Type{Active{RT}},
                                        f::EnzymeCore.Const{typeof(Base.identity)},
                                        op::EnzymeCore.Const{typeof(Base.add_sum)},
                                        A::EnzymeCore.Annotation{<:AnyCuArray{T}}; dims::D, init) where {RT, T, D}
    primal = if EnzymeRules.needs_primal(config)
        ofn.val(f.val, op.val, A.val; dims, init)
    else
        nothing
    end
    
    shadow = if EnzymeRules.needs_shadow(config)
        A.dval
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(GPUArrays._mapreduce)},
                                        dres::Active{RT},
                                        tape,
                                        f::EnzymeCore.Const{typeof(Base.identity)},
                                        op::EnzymeCore.Const{typeof(Base.add_sum)},
                                        A::EnzymeCore.Annotation{<:AnyCuArray{T}}; dims::D, init) where {RT, T, D}

    # `Ref` so that the cotangent is broadcast as a scalar; for a non-scalar
    # element type like `SVector` it would otherwise broadcast over its own axes
    if A isa Duplicated || A isa DuplicatedNoNeed
        A.dval .+= Ref(dres.val)
    elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.batch_width(A))) do i
            Base.@_inline_meta
            A.dval[i] .+= Ref(dres.val)
            nothing
        end
    end

    return (nothing, nothing, nothing)
end

end # module

