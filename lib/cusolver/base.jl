# wrappers of low-level functionality

function cusolverGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cusolverGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cusolverGetProperty(CUDA.MAJOR_VERSION),
                          cusolverGetProperty(CUDA.MINOR_VERSION),
                          cusolverGetProperty(CUDA.PATCH_LEVEL))

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

function Base.convert(::Type{cusolverEigRange_t}, range::Char)
    if range == 'A'
        CUSOLVER_EIG_RANGE_ALL
    elseif range == 'V'
        CUSOLVER_EIG_RANGE_V
    elseif range == 'I'
        CUSOLVER_EIG_RANGE_I
    else
        throw(ArgumentError("Unknown eigenvalue solver range $range."))
    end
end

function Base.convert(::Type{cusolverStorevMode_t}, storev::Char)
    if storev == 'C'
        CUBLAS_STOREV_COLUMNWISE
    elseif storev == 'R'
        CUBLAS_STOREV_ROWWISE
    else
        throw(ArgumentError("Unknown storage mode $storev."))
    end
end

function Base.convert(::Type{cusolverDirectMode_t}, direct::Char)
    if direct == 'F'
        CUBLAS_DIRECT_FORWARD
    elseif direct == 'B'
        CUBLAS_DIRECT_BACKWARD
    else
        throw(ArgumentError("Unknown direction mode $direct."))
    end
end
