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

function Base.convert(::Type{cusolverIRSRefinement_t}, irs::String)
    if irs == "NOT_SET"
        CUSOLVER_IRS_REFINE_NOT_SET
    elseif irs == "NONE"
        CUSOLVER_IRS_REFINE_NONE
    elseif irs == "CLASSICAL"
        CUSOLVER_IRS_REFINE_CLASSICAL
    elseif "CLASSICAL_GMRES"
        CUSOLVER_IRS_REFINE_CLASSICAL_GMRES
    elseif "GMRES"
        CUSOLVER_IRS_REFINE_GMRES
    elseif "GMRES_GMRES"
        CUSOLVER_IRS_REFINE_GMRES_GMRES
    elseif "GMRES_NOPCOND"
        CUSOLVER_IRS_REFINE_GMRES_NOPCOND
    else
        throw(ArgumentError("Unknown iterative refinement solver $irs."))
    end
end

function Base.convert(::Type{cusolverPrecType_t}, T::String)
    if T == "R_16F"
        return CUSOLVER_R_16F
    elseif T == "R_16BF"
        return CUSOLVER_R_16BF
    elseif T == "R_TF32"
        return CUSOLVER_R_TF32
    elseif T == "R_32F"
        return CUSOLVER_R_32F
    elseif T == "R_64F"
        return CUSOLVER_R_64F
    elseif T == "C_16F"
        return CUSOLVER_C_16F
    elseif T == "C_16BF"
        return CUSOLVER_C_16BF
    elseif T == "C_TF32"
        return CUSOLVER_C_TF32
    elseif T == "C_32F"
        return CUSOLVER_C_32F
    elseif T == "C_64F"
        return CUSOLVER_C_64F
    else
        throw(ArgumentError("cusolverPrecType_t equivalent for input type $T does not exist!"))
    end
end

function Base.convert(::Type{cusolverPrecType_t}, T::DataType)
    if T === Float32
        return CUSOLVER_R_32F
    elseif T === Float64
        return CUSOLVER_R_64F
    elseif T === Complex{Float32}
        return CUSOLVER_C_32F
    elseif T === Complex{Float64}
        return CUSOLVER_C_64F
    else
        throw(ArgumentError("cusolverPrecType_t equivalent for input type $T does not exist!"))
    end
end
