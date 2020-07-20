# wrappers of low-level functionality

@memoize function cusolverGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cusolverGetProperty(property, value_ref)
  value_ref[]
end

@memoize version() = VersionNumber(cusolverGetProperty(CUDA.MAJOR_VERSION),
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
