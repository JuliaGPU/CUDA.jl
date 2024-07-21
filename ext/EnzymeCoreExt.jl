# compatibility with EnzymeCore

module EnzymeCoreExt

using CUDA
import CUDA: GPUCompiler, CUDABackend

if isdefined(Base, :get_extension)
    using EnzymeCore
    using EnzymeCore.EnzymeRules
else
    using ..EnzymeCore
    using ..EnzymeCore.EnzymeRules
end

function EnzymeCore.EnzymeRules.inactive(::typeof(CUDA.CUBLAS.handle))
    return nothing
end
function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(CUDA.CUBLAS.version))
    return nothing
end
function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(CUDA.context!), args...)
    return nothing
end
function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(CUDA.is_pinned), args...)
    return nothing
end

function EnzymeCore.compiler_job_from_backend(::CUDABackend, @nospecialize(F::Type), @nospecialize(TT::Type))
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, CUDA.compiler_config(CUDA.device()))
end

function metaf(fn, args::Vararg{Any, N}) where N
    EnzymeCore.autodiff_deferred(Forward, fn, Const, args...)
    nothing
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{typeof(cufunction)},
                                        ::Type{<:Duplicated}, f::Const{F},
                                        tt::Const{TT}; kwargs...) where {F,TT}
    res = ofn.val(f.val, tt.val; kwargs...)
    return Duplicated(res, res)
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{typeof(cufunction)},
                                        ::Type{BatchDuplicated{T,N}}, f::Const{F},
                                        tt::Const{TT}; kwargs...) where {F,TT,T,N}
    res = ofn.val(f.val, tt.val; kwargs...)
    return BatchDuplicated(res, ntuple(Val(N)) do _
        res
    end)
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{typeof(cudaconvert)},
                                        ::Type{RT}, x::IT) where {RT, IT}
    if RT <: Duplicated
        RT(ofn.val(x.val), ofn.val(x.dval))
    elseif RT <: Const
        ofn.val(x.val)::eltype(RT)
    elseif RT <: DuplicatedNoNeed
        ofn.val(x.val)::eltype(RT)
    else
        tup = ntuple(Val(EnzymeCore.batch_size(RT))) do i
            Base.@_inline_meta
            ofn.val(x.dval[i])::eltype(RT)
        end
        if RT <: BatchDuplicated
            BatchDuplicated(ofv.val(x.val), tup)
        else
            tup
        end
    end
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{UndefInitializer}, args...) where {CT <: CuArray, RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end
    if RT <: Duplicated
        shadow = ofn.val(uval.val, primargs...)::CT
        fill!(shadow, 0)
        Duplicated(ofn.val(uval.val, primargs...), shadow)
    elseif RT <: Const
        ofn.val(uval.val, primargs...)
    elseif RT <: DuplicatedNoNeed
        shadow = ofn.val(uval.val, primargs...)::CT
        fill!(shadow, 0)
        shadow::CT
    else
        tup = ntuple(Val(EnzymeCore.batch_size(RT))) do i
            Base.@_inline_meta
            shadow = ofn.val(uval.val, primargs...)::CT
            fill!(shadow, 0)
            shadow::CT
        end
        if RT <: BatchDuplicated
            BatchDuplicated(ofv.val(uval.val), tup)
        else
            tup
        end
    end
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDA.DataRef, RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end
    if RT <: Duplicated
        shadow = ofn.val(uval.val, primargs...; kwargs...)
        Duplicated(ofn.val(uval.dval, primargs...; kwargs...), shadow)
    elseif RT <: Const
        ofn.val(uval.val, primargs...; kwargs...)
    elseif RT <: DuplicatedNoNeed
        ofn.val(uval.dval, primargs...; kwargs...)
    else
        tup = ntuple(Val(EnzymeCore.batch_size(RT))) do i
            Base.@_inline_meta
            shadow = ofn.val(uval.dval[i], primargs...; kwargs...)
        end
        if RT <: BatchDuplicated
            BatchDuplicated(ofv.val(uval.val), tup)
        else
            tup
        end
    end
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{typeof(synchronize)},
                                        ::Type{RT}, args::Vararg{EnzymeCore.Annotation, N}; kwargs...) where {RT, N}
    pargs = ntuple(Val(N)) do i
        Base.@_inline_meta
        args[i].val
    end
    res = ofn.val(pargs...; kwargs...)

    if RT <: Duplicated
        return Duplicated(res, res)
    elseif RT <: Const
        return res
    elseif RT <: DuplicatedNoNeed
        return res
    else
        tup = ntuple(Val(EnzymeCore.batch_size(RT))) do i
            Base.@_inline_meta
            res
        end
        if RT <: BatchDuplicated
            return BatchDuplicated(res, tup)
        else
            return tup
        end
    end
end

function EnzymeCore.EnzymeRules.forward(ofn::EnzymeCore.Annotation{CUDA.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, args...;
                                        kwargs...) where {F,TT}

    GC.@preserve args begin
        args = ((cudaconvert(a) for a in args)...,)
        T2 = (F, (typeof(a) for a in args)...)
        TT2 = Tuple{T2...}
        cuf = cufunction(metaf, TT2)
        res = cuf(ofn.val.f, args...; kwargs...)
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
    return EnzymeRules.AugmentedReturn{(EnzymeRules.needs_primal(config) ? CT : Nothing), (EnzymeRules.needs_shadow(config) ? (EnzymeRules.width(config) == 1 ? CT : NTuple{EnzymeRules.width(config), CT}) : Nothing), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::EnzymeCore.Const{typeof(cufunction)},::Type{RT}, subtape, f, tt; kwargs...) where RT
    return (nothing, nothing)
end

function meta_augf(f, tape::CuDeviceArray{TapeType}, ::Val{ModifiedBetween}, args::Vararg{Any, N}) where {N, ModifiedBetween, TapeType}
    forward, _ = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        map(typeof, args)...,
    )

    idx = 0
    # idx *= gridDim().x
    idx += blockIdx().x
    
    idx *= gridDim().y
    idx += blockIdx().y
    
    idx *= gridDim().z
    idx += blockIdx().z
    
    idx *= blockDim().x
    idx += threadIdx().x
    
    idx *= blockDim().y
    idx += threadIdx().y
   
    idx *= blockDim().z
    idx += threadIdx().z

    @inbounds tape[idx] = forward(Const(f), args...)[1]
    nothing
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::EnzymeCore.Annotation{CUDA.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, args0...;
                                        threads::CuDim=1, blocks::CuDim=1, kwargs...) where {F,TT}
    args = ((cudaconvert(arg) for arg in args0)...,)
    ModifiedBetween = overwritten(config)
    TapeType = EnzymeCore.tape_type(
        EnzymeCore.compiler_job_from_backend(CUDABackend(), typeof(Base.identity), Tuple{Float64}),
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
        Const{F},
        Const{Nothing},
        map(typeof, args)...,
    )
    threads = CuDim3(threads)
    blocks = CuDim3(blocks)
    subtape = CuArray{TapeType}(undef, blocks.x*blocks.y*blocks.z*threads.x*threads.y*threads.z)

    GC.@preserve args0 subtape, begin
        subtape2 = cudaconvert(subtape)
        T2 = (F, typeof(subtape2), Val{ModifiedBetween},   (typeof(a) for a in args)...)
        TT2 = Tuple{T2...}
        cuf = cufunction(meta_augf, TT2)
        res = cuf(ofn.val.f, subtape2, Val(ModifiedBetween), args...; threads=(threads.x, threads.y, threads.z), blocks=(blocks.x, blocks.y, blocks.z), kwargs...)
    end

    return AugmentedReturn{Nothing,Nothing,CuArray}(nothing, nothing, subtape)
end

function meta_revf(f, tape::CuDeviceArray{TapeType}, ::Val{ModifiedBetween},  args::Vararg{Any, N}) where {N, ModifiedBetween, TapeType}
    _, reverse = EnzymeCore.autodiff_deferred_thunk(
        EnzymeCore.compiler_job_from_backend(CUDABackend(), typeof(Base.identity), Tuple{Float64}),
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        map(typeof, args)...,
    )

    idx = 0
    # idx *= gridDim().x
    idx += blockIdx().x
    
    idx *= gridDim().y
    idx += blockIdx().y
    
    idx *= gridDim().z
    idx += blockIdx().z
    
    idx *= blockDim().x
    idx += threadIdx().x
    
    idx *= blockDim().y
    idx += threadIdx().y
   
    idx *= blockDim().z
    idx += threadIdx().z
    reverse(Const(f), args..., @inbounds tape[idx])
    nothing
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::EnzymeCore.Annotation{CUDA.HostKernel{F,TT}},
                                        ::Type{Const{Nothing}}, subtape, args0...;
                                        threads::CuDim=1, blocks::CuDim=1, kwargs...) where {F,TT}
    args = ((cudaconvert(arg) for arg in args0)...,)
    ModifiedBetween = overwritten(config)
    TapeType = EnzymeCore.tape_type(
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
        Const{F},
        Const{Nothing},
        map(typeof, args)...,
    )
    threads = CuDim3(threads)
    blocks = CuDim3(blocks)

    GC.@preserve args0 subtape, begin
        subtape2 = cudaconvert(subtape)
        T2 = (F, typeof(subtape2), Val{ModifiedBetween}, (typeof(a) for a in args)...)
        TT2 = Tuple{T2...}
        cuf = cufunction(meta_revf, TT2)
        res = cuf(ofn.val.f, subtape2, Val(ModifiedBetween), args...; threads=(threads.x, threads.y, threads.z), blocks=(blocks.x, blocks.y, blocks.z), kwargs...)
    end

    return ntuple(Val(length(args0))) do i
        Base.@_inline_meta
        nothing
    end
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{typeof(Base.fill!)}, ::Type{RT}, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x) where {RT, T <: CUDA.MemsetCompatTypes}
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

    if RT <: Duplicated
        return A
    elseif RT <: Const
        return A.val
    elseif RT <: DuplicatedNoNeed
        return A.dval
    elseif RT <: BatchDuplicated
        return A
    else
        return A.dval
    end
end


function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{typeof(Base.fill!)}, ::Type{RT}, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x) where {RT, T <: CUDA.MemsetCompatTypes}
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

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{typeof(Base.fill!)}, ::Type{RT}, tape, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x::EnzymeCore.Annotation{T2}) where {RT, T <: CUDA.MemsetCompatTypes, T2}
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
            fill!(subshadow, 0)
            subshadow
        else
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              subshadow = ofn.val(uval.val, primargs...)::CT
              fill!(subshadow, 0)
              subshadow
          end
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn{(EnzymeRules.needs_primal(config) ? eltype(RT) : Nothing), (EnzymeRules.needs_shadow(config) ? (EnzymeRules.width(config) == 1 ? eltype(RT) : NTuple{EnzymeRules.width(config), eltype(RT)}) : Nothing), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape, A::EnzymeCore.Annotation{UndefInitializer}, args::Vararg{EnzymeCore.Annotation, N}) where {CT <: CuArray, RT, N}
    ntuple(Val(N+1)) do i
          Base.@_inline_meta
          nothing
    end
end

function EnzymeCore.EnzymeRules.augmented_primal(config, ofn::Const{Type{CT}}, ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDA.DataRef, RT}
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
    return EnzymeRules.AugmentedReturn{(EnzymeRules.needs_primal(config) ? eltype(RT) : Nothing), (EnzymeRules.needs_shadow(config) ? (EnzymeRules.width(config) == 1 ? eltype(RT) : NTuple{EnzymeRules.width(config), eltype(RT)}) : Nothing), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, ofn::Const{Type{CT}}, ::Type{RT}, tape, A::EnzymeCore.Annotation{DR}, args::Vararg{EnzymeCore.Annotation, N}; kwargs...) where {CT <: CuArray, DR <: CUDA.DataRef, RT, N}
    ntuple(Val(N+1)) do i
          Base.@_inline_meta
          nothing
    end
end

function EnzymeCore.EnzymeRules.noalias(::Type{CT}, ::UndefInitializer, args...) where {CT <: CuArray}
    return nothing
end

end # module

