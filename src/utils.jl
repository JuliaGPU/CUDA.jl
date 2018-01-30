# take some kwargs off of a splatted list
@inline function take_kwargs(wanted_kws...; kwargs...)
    return _take_kwargs(Val(wanted_kws); kwargs...)
end

@generated function _take_kwargs(::Val{wanted_kws}; kwargs...) where {wanted_kws}
    wanted_kwargs, remaining_kwargs = gen_take_kwargs(kwargs, wanted_kws...)
    quote
        ($wanted_kwargs...,), ($remaining_kwargs...,)
    end
end

function gen_take_kwargs(kwargs, wanted_kws...)
    nt = kwargs.parameters[4]

    provided_kws = Base._nt_names(nt)::NTuple{N,Symbol} where {N}
    wanted_kws = Symbol[provided_kws...] ∩ wanted_kws # JuliaLang/julia#25801
    remaining_kws = setdiff(provided_kws, wanted_kws)

    get_kwargs(kws) = Tuple(:($kw=kwargs[$(QuoteNode(kw))]) for kw in kws)
    get_kwargs(wanted_kws), get_kwargs(remaining_kws)
end
