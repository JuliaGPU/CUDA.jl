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
using GPUArrays

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
            ofn.val(x.dval)::eltype(RT)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                ofn.val(x.dval[i])::eltype(RT)
            end
        end
    elseif EnzymeRules.needs_primal(config)
        ofn.val(uval.val)::eltype(RT)
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
        if EnzymeRules.width(config) == 1
            ofn.val(x.dval)
        else
          ntuple(Val(EnzymeRules.width(config))) do i
              Base.@_inline_meta
              ofn.val(x.dval[i])
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
            fill!(shadow, 0)
            Duplicated(ofn.val(uval.val, primargs...), shadow)
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                shadow = ofn.val(uval.val, primargs...)::CT
                fill!(shadow, 0)
                shadow::CT
            end
            BatchDuplicated(ofn.val(uval.val, primargs...), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = ofn.val(uval.val, primargs...)::CT
            fill!(shadow, 0)
            shadow
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                shadow = ofn.val(uval.val, primargs...)::CT
                fill!(shadow, 0)
                shadow::CT
            end
            tup
        end
    elseif EnzymeRules.needs_primal(config)
        ofn.val(uval.val, primargs...)
    else
        nothing
    end
end

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{Type{CT}},
        ::Type{RT}, uval::EnzymeCore.Annotation{DR}, args...; kwargs...) where {CT <: CuArray, DR <: CUDA.DataRef, RT}
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

function EnzymeCore.EnzymeRules.forward(config, ofn::EnzymeCore.Annotation{CUDA.HostKernel{F,TT}},
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
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
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

    GC.@preserve args subtape, begin
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
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
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

function EnzymeCore.EnzymeRules.forward(config, ofn::Const{typeof(Base.fill!)}, ::Type{RT}, A::EnzymeCore.Annotation{<:DenseCuArray{T}}, x) where {RT, T <: CUDA.MemsetCompatTypes}
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
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
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
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT), EnzymeRules.shadow_type(config, RT), Nothing}(primal, shadow, nothing)
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
                                        R::EnzymeCore.Annotation{<:AnyCuArray{T}}, A; init) where {RT, T<:AbstractFloat}
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
                                        R::EnzymeCore.Annotation{<:AnyCuArray{T}}, A; init) where {RT, T<:AbstractFloat}

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
                                        A::EnzymeCore.Annotation{<:AnyCuArray{T}}; dims::D, init) where {RT, T<:AbstractFloat, D}
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
                                        A::EnzymeCore.Annotation{<:AnyCuArray{T}}; dims::D, init) where {RT, T<:AbstractFloat, D}

    if A isa Duplicated || A isa DuplicatedNoNeed
        A.dval .+= dres.val
    elseif A isa BatchDuplicated || A isa BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.batch_width(A))) do i
            Base.@_inline_meta
            A.dval[i] .+= dres.val
            nothing
        end
    end

    return (nothing, nothing, nothing)
end

end # module

