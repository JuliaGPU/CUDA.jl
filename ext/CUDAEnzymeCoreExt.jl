module CUDAEnzymeCoreExt

using CUDA

if isdefined(Base, :get_extension)
    using EnzymeCore
    using EnzymeCore.EnzymeRules
else
    using ..EnzymeCore
    using ..EnzymeCore.EnzymeRules
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

end
