function Base.convert(::Type{cusolverEigType_t}, typ::Int)
    if typ == 1
        CUSOLVER_EIG_TYPE_1
    elseif typ == 2
        CUSOLVER_EIG_TYPE_2
    elseif typ == 3
        CUSOLVER_EIG_TYPE_3
    else
        throw(ArgumentError("Unknown eigenvalue solver type $typ."))
    end
end

function Base.convert(::Type{cusolverEigMode_t}, jobz::Char)
    if jobz == 'N'
        CUSOLVER_EIG_MODE_NOVECTOR
    elseif jobz == 'V'
        CUSOLVER_EIG_MODE_VECTOR
    else
        throw(ArgumentError("Unknown eigenvalue solver mode $jobz."))
    end
end
