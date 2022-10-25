using CEnum

# CURAND uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CURAND_STATUS_ALLOCATION_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CURANDError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CURAND_STATUS_ALLOCATION_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CURAND_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end


@cenum curandStatus::UInt32 begin
    CURAND_STATUS_SUCCESS = 0
    CURAND_STATUS_VERSION_MISMATCH = 100
    CURAND_STATUS_NOT_INITIALIZED = 101
    CURAND_STATUS_ALLOCATION_FAILED = 102
    CURAND_STATUS_TYPE_ERROR = 103
    CURAND_STATUS_OUT_OF_RANGE = 104
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
    CURAND_STATUS_LAUNCH_FAILURE = 201
    CURAND_STATUS_PREEXISTING_FAILURE = 202
    CURAND_STATUS_INITIALIZATION_FAILED = 203
    CURAND_STATUS_ARCH_MISMATCH = 204
    CURAND_STATUS_INTERNAL_ERROR = 999
end

const curandStatus_t = curandStatus

@cenum curandRngType::UInt32 begin
    CURAND_RNG_TEST = 0
    CURAND_RNG_PSEUDO_DEFAULT = 100
    CURAND_RNG_PSEUDO_XORWOW = 101
    CURAND_RNG_PSEUDO_MRG32K3A = 121
    CURAND_RNG_PSEUDO_MTGP32 = 141
    CURAND_RNG_PSEUDO_MT19937 = 142
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161
    CURAND_RNG_QUASI_DEFAULT = 200
    CURAND_RNG_QUASI_SOBOL32 = 201
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
    CURAND_RNG_QUASI_SOBOL64 = 203
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
end

const curandRngType_t = curandRngType

@cenum curandOrdering::UInt32 begin
    CURAND_ORDERING_PSEUDO_BEST = 100
    CURAND_ORDERING_PSEUDO_DEFAULT = 101
    CURAND_ORDERING_PSEUDO_SEEDED = 102
    CURAND_ORDERING_PSEUDO_LEGACY = 103
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104
    CURAND_ORDERING_QUASI_DEFAULT = 201
end

const curandOrdering_t = curandOrdering

@cenum curandDirectionVectorSet::UInt32 begin
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
end

const curandDirectionVectorSet_t = curandDirectionVectorSet

const curandDirectionVectors32_t = NTuple{32, Cuint}

const curandDirectionVectors64_t = NTuple{64, Culonglong}

mutable struct curandGenerator_st end

const curandGenerator_t = Ptr{curandGenerator_st}

const curandDistribution_st = Cdouble

const curandDistribution_t = Ptr{curandDistribution_st}

mutable struct curandDistributionShift_st end

const curandDistributionShift_t = Ptr{curandDistributionShift_st}

mutable struct curandDistributionM2Shift_st end

const curandDistributionM2Shift_t = Ptr{curandDistributionM2Shift_st}

mutable struct curandHistogramM2_st end

const curandHistogramM2_t = Ptr{curandHistogramM2_st}

const curandHistogramM2K_st = Cuint

const curandHistogramM2K_t = Ptr{curandHistogramM2K_st}

const curandHistogramM2V_st = curandDistribution_st

const curandHistogramM2V_t = Ptr{curandHistogramM2V_st}

mutable struct curandDiscreteDistribution_st end

const curandDiscreteDistribution_t = Ptr{curandDiscreteDistribution_st}

@cenum curandMethod::UInt32 begin
    CURAND_CHOOSE_BEST = 0
    CURAND_ITR = 1
    CURAND_KNUTH = 2
    CURAND_HITR = 3
    CURAND_M1 = 4
    CURAND_M2 = 5
    CURAND_BINARY_SEARCH = 6
    CURAND_DISCRETE_GAUSS = 7
    CURAND_REJECTION = 8
    CURAND_DEVICE_API = 9
    CURAND_FAST_REJECTION = 10
    CURAND_3RD = 11
    CURAND_DEFINITION = 12
    CURAND_POISSON = 13
end

const curandMethod_t = curandMethod

@checked function curandCreateGenerator(generator, rng_type)
        initialize_context()
        ccall((:curandCreateGenerator, libcurand), curandStatus_t, (Ptr{curandGenerator_t}, curandRngType_t), generator, rng_type)
    end

@checked function curandCreateGeneratorHost(generator, rng_type)
        initialize_context()
        ccall((:curandCreateGeneratorHost, libcurand), curandStatus_t, (Ptr{curandGenerator_t}, curandRngType_t), generator, rng_type)
    end

@checked function curandDestroyGenerator(generator)
        initialize_context()
        ccall((:curandDestroyGenerator, libcurand), curandStatus_t, (curandGenerator_t,), generator)
    end

@checked function curandGetVersion(version)
        ccall((:curandGetVersion, libcurand), curandStatus_t, (Ptr{Cint},), version)
    end

@checked function curandGetProperty(type, value)
        ccall((:curandGetProperty, libcurand), curandStatus_t, (libraryPropertyType, Ptr{Cint}), type, value)
    end

@checked function curandSetStream(generator, stream)
        initialize_context()
        ccall((:curandSetStream, libcurand), curandStatus_t, (curandGenerator_t, cudaStream_t), generator, stream)
    end

@checked function curandSetPseudoRandomGeneratorSeed(generator, seed)
        initialize_context()
        ccall((:curandSetPseudoRandomGeneratorSeed, libcurand), curandStatus_t, (curandGenerator_t, Culonglong), generator, seed)
    end

@checked function curandSetGeneratorOffset(generator, offset)
        initialize_context()
        ccall((:curandSetGeneratorOffset, libcurand), curandStatus_t, (curandGenerator_t, Culonglong), generator, offset)
    end

@checked function curandSetGeneratorOrdering(generator, order)
        initialize_context()
        ccall((:curandSetGeneratorOrdering, libcurand), curandStatus_t, (curandGenerator_t, curandOrdering_t), generator, order)
    end

@checked function curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions)
        initialize_context()
        ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand), curandStatus_t, (curandGenerator_t, Cuint), generator, num_dimensions)
    end

@checked function curandGenerate(generator, outputPtr, num)
        initialize_context()
        ccall((:curandGenerate, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{UInt32}, Csize_t), generator, outputPtr, num)
    end

@checked function curandGenerateLongLong(generator, outputPtr, num)
        initialize_context()
        ccall((:curandGenerateLongLong, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Culonglong}, Csize_t), generator, outputPtr, num)
    end

@checked function curandGenerateUniform(generator, outputPtr, num)
        initialize_context()
        ccall((:curandGenerateUniform, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Cfloat}, Csize_t), generator, outputPtr, num)
    end

@checked function curandGenerateUniformDouble(generator, outputPtr, num)
        initialize_context()
        ccall((:curandGenerateUniformDouble, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Cdouble}, Csize_t), generator, outputPtr, num)
    end

@checked function curandGenerateNormal(generator, outputPtr, n, mean, stddev)
        initialize_context()
        ccall((:curandGenerateNormal, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat), generator, outputPtr, n, mean, stddev)
    end

@checked function curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev)
        initialize_context()
        ccall((:curandGenerateNormalDouble, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble), generator, outputPtr, n, mean, stddev)
    end

@checked function curandGenerateLogNormal(generator, outputPtr, n, mean, stddev)
        initialize_context()
        ccall((:curandGenerateLogNormal, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat), generator, outputPtr, n, mean, stddev)
    end

@checked function curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev)
        initialize_context()
        ccall((:curandGenerateLogNormalDouble, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble), generator, outputPtr, n, mean, stddev)
    end

@checked function curandCreatePoissonDistribution(lambda, discrete_distribution)
        initialize_context()
        ccall((:curandCreatePoissonDistribution, libcurand), curandStatus_t, (Cdouble, Ptr{curandDiscreteDistribution_t}), lambda, discrete_distribution)
    end

@checked function curandDestroyDistribution(discrete_distribution)
        initialize_context()
        ccall((:curandDestroyDistribution, libcurand), curandStatus_t, (curandDiscreteDistribution_t,), discrete_distribution)
    end

@checked function curandGeneratePoisson(generator, outputPtr, n, lambda)
        initialize_context()
        ccall((:curandGeneratePoisson, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cdouble), generator, outputPtr, n, lambda)
    end

@checked function curandGeneratePoissonMethod(generator, outputPtr, n, lambda, method)
        initialize_context()
        ccall((:curandGeneratePoissonMethod, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cdouble, curandMethod_t), generator, outputPtr, n, lambda, method)
    end

@checked function curandGenerateBinomial(generator, outputPtr, num, n, p)
        initialize_context()
        ccall((:curandGenerateBinomial, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cuint, Cdouble), generator, outputPtr, num, n, p)
    end

@checked function curandGenerateBinomialMethod(generator, outputPtr, num, n, p, method)
        initialize_context()
        ccall((:curandGenerateBinomialMethod, libcurand), curandStatus_t, (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cuint, Cdouble, curandMethod_t), generator, outputPtr, num, n, p, method)
    end

@checked function curandGenerateSeeds(generator)
        initialize_context()
        ccall((:curandGenerateSeeds, libcurand), curandStatus_t, (curandGenerator_t,), generator)
    end

@checked function curandGetDirectionVectors32(vectors, set)
        initialize_context()
        ccall((:curandGetDirectionVectors32, libcurand), curandStatus_t, (Ptr{Ptr{curandDirectionVectors32_t}}, curandDirectionVectorSet_t), vectors, set)
    end

@checked function curandGetScrambleConstants32(constants)
        initialize_context()
        ccall((:curandGetScrambleConstants32, libcurand), curandStatus_t, (Ptr{Ptr{Cuint}},), constants)
    end

@checked function curandGetDirectionVectors64(vectors, set)
        initialize_context()
        ccall((:curandGetDirectionVectors64, libcurand), curandStatus_t, (Ptr{Ptr{curandDirectionVectors64_t}}, curandDirectionVectorSet_t), vectors, set)
    end

@checked function curandGetScrambleConstants64(constants)
        initialize_context()
        ccall((:curandGetScrambleConstants64, libcurand), curandStatus_t, (Ptr{Ptr{Culonglong}},), constants)
    end

