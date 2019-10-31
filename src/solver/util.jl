# convert Char {N,V} to cusolverEigMode_t
function cusolverjob(jobz::Char)
    if jobz == 'N'
        return CUSOLVER_EIG_MODE_NOVECTOR
    end
    if jobz == 'V'
        return CUSOLVER_EIG_MODE_VECTOR
    end
    throw(ArgumentError("unknown cusolver eigmode $jobz."))
end
