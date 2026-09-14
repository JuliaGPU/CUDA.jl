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

function EnzymeCore.EnzymeRules.forward(config, ofn::EnzymeCore.Annotation{CUDACore.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, args...;
                                        kwargs...) where {F,TT}

    GC.@preserve args begin
        args = ((cudaconvert(a) for a in args)...,)
	T2 = (typeof(config), F, (typeof(a) for a in args)...)
        TT2 = Tuple{T2...}
        cuf = cufunction(metaf, TT2)
        res = cuf(config, ofn.val.f, args...; kwargs...)
    end

    return nothing
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

function meta_augf(config, f, tape::CuDeviceArray{TapeType}, args::Vararg{Any, N}) where {N, TapeType}
    ModifiedBetween = overwritten(config)
    forward, _ = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        map(typeof, args)...,
    )

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
    idx += 1

    @inbounds tape[idx] = forward(Const(f), args...)[1]
    nothing
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::EnzymeCore.Annotation{CUDACore.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, args0...;
                                        threads::CuDim=1, blocks::CuDim=1, kwargs...) where {F,TT}
    args = ((cudaconvert(arg) for arg in args0)...,)
    ModifiedBetween = overwritten(config)
    TapeType = EnzymeCore.tape_type(
        EnzymeCore.compiler_job_from_backend(CUDABackend(), typeof(Base.identity), Tuple{Float64}),
	ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        Const{F},
        Const{Nothing},
        map(typeof, args)...,
    )
    threads = CuDim3(threads)
    blocks = CuDim3(blocks)
    subtape = CuArray{TapeType}(undef, blocks.x*blocks.y*blocks.z*threads.x*threads.y*threads.z)

    GC.@preserve args subtape, begin
        subtape2 = cudaconvert(subtape)
	T2 = (typeof(config), F, typeof(subtape2), (typeof(a) for a in args)...)
        TT2 = Tuple{T2...}
        cuf = cufunction(meta_augf, TT2)
        res = cuf(config, ofn.val.f, subtape2, args...; threads=(threads.x, threads.y, threads.z), blocks=(blocks.x, blocks.y, blocks.z), kwargs...)
    end

    return AugmentedReturn{Nothing,Nothing,CuArray}(nothing, nothing, subtape)
end

function meta_revf(config, f, tape::CuDeviceArray{TapeType}, args::Vararg{Any, N}) where {N, TapeType}
    ModifiedBetween = overwritten(config)
    _, reverse = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        map(typeof, args)...,
    )

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
    idx += 1
    reverse(Const(f), args..., @inbounds tape[idx])
    nothing
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::EnzymeCore.Annotation{CUDACore.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, subtape, args0...;
                                        threads::CuDim=1, blocks::CuDim=1, kwargs...) where {F,TT}
    args = ((cudaconvert(arg) for arg in args0)...,)
    ModifiedBetween = overwritten(config)
    TapeType = EnzymeCore.tape_type(
        ReverseSplitModified(EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config), Val(ModifiedBetween)),
        Const{F},
        Const{Nothing},
        map(typeof, args)...,
    )
    threads = CuDim3(threads)
    blocks = CuDim3(blocks)

    GC.@preserve args0 subtape, begin
        subtape2 = cudaconvert(subtape)
	T2 = (typeof(config), F, typeof(subtape2), (typeof(a) for a in args)...)
        TT2 = Tuple{T2...}
        cuf = cufunction(meta_revf, TT2)
        res = cuf(config, ofn.val.f, subtape2, args...; threads=(threads.x, threads.y, threads.z), blocks=(blocks.x, blocks.y, blocks.z), kwargs...)
    end

    return ntuple(Val(length(args0))) do i
        Base.@_inline_meta
        nothing
    end
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

# `make_zero`/`make_zero!` for GPU arrays, the `Base.fill!` rules and the rules
# for `GPUArrays.mapreducedim!`/`GPUArrays._mapreduce` used to live here, keyed on
# `DenseCuArray`/`AnyCuArray`. Nothing in them was CUDA specific, so they moved to
# Enzyme's GPUArraysCore and GPUArrays extensions, keyed on `AbstractGPUArray`,
# where every backend gets them. The `fill!` rules covered only
# `MemsetCompatTypes` here and now cover every float, complex and integer element
# type. Requires Enzyme >= 0.13.204.

end # module

