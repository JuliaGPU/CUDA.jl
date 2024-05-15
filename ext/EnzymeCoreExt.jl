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
        return Duplicated(ofn.val(x.val), ofn.val(x.dval))
    elseif RT <: Const
        return ofn.val(x.val)
    elseif RT <: DuplicatedNoNeed
        return ofn.val(x.val)
    else
        tup = ntuple(Val(EnzymeCore.batch_size(RT))) do i
            Base.@_inline_meta
            ofn.val(x.dval[i])
        end
        if RT <: BatchDuplicated
            return BatchDuplicated(ofv.val(x.val), tup)
        else
            return tup
        end
    end
end

function EnzymeCore.EnzymeRules.forward(ofn::Const{typeof(synchronize)},
                                        ::Type{RT}, args::NTuple{N, EnzymeCore.Annotation}; kwargs...) where {RT, N}
    pargs = ntuple(Val(N)) do i
        Base.@_inline_meta
        args.val
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

end # module

