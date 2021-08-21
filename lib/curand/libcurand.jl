# Julia wrapper for header: curand.h
# Automatically generated using Clang.jl

@checked function curandCreateGenerator(generator, rng_type)
    initialize_api()
    ccall((:curandCreateGenerator, libcurand()), curandStatus_t,
                   (Ptr{curandGenerator_t}, curandRngType_t),
                   generator, rng_type)
end

@checked function curandCreateGeneratorHost(generator, rng_type)
    initialize_api()
    ccall((:curandCreateGeneratorHost, libcurand()), curandStatus_t,
                   (Ptr{curandGenerator_t}, curandRngType_t),
                   generator, rng_type)
end

@checked function curandDestroyGenerator(generator)
    initialize_api()
    ccall((:curandDestroyGenerator, libcurand()), curandStatus_t,
                   (curandGenerator_t,),
                   generator)
end

@checked function curandGetVersion(version)
    ccall((:curandGetVersion, libcurand()), curandStatus_t,
                   (Ptr{Cint},),
                   version)
end

@checked function curandGetProperty(type, value)
    ccall((:curandGetProperty, libcurand()), curandStatus_t,
                   (libraryPropertyType, Ptr{Cint}),
                   type, value)
end

@checked function curandSetStream(generator, stream)
    initialize_api()
    ccall((:curandSetStream, libcurand()), curandStatus_t,
                   (curandGenerator_t, CUstream),
                   generator, stream)
end

@checked function curandSetPseudoRandomGeneratorSeed(generator, seed)
    initialize_api()
    ccall((:curandSetPseudoRandomGeneratorSeed, libcurand()), curandStatus_t,
                   (curandGenerator_t, Culonglong),
                   generator, seed)
end

@checked function curandSetGeneratorOffset(generator, offset)
    initialize_api()
    ccall((:curandSetGeneratorOffset, libcurand()), curandStatus_t,
                   (curandGenerator_t, Culonglong),
                   generator, offset)
end

@checked function curandSetGeneratorOrdering(generator, order)
    initialize_api()
    ccall((:curandSetGeneratorOrdering, libcurand()), curandStatus_t,
                   (curandGenerator_t, curandOrdering_t),
                   generator, order)
end

@checked function curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions)
    initialize_api()
    ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand()), curandStatus_t,
                   (curandGenerator_t, UInt32),
                   generator, num_dimensions)
end

@checked function curandGenerate(generator, outputPtr, num)
    initialize_api()
    ccall((:curandGenerate, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{UInt32}, Csize_t),
                   generator, outputPtr, num)
end

@checked function curandGenerateLongLong(generator, outputPtr, num)
    initialize_api()
    ccall((:curandGenerateLongLong, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Culonglong}, Csize_t),
                   generator, outputPtr, num)
end

@checked function curandGenerateUniform(generator, outputPtr, num)
    initialize_api()
    ccall((:curandGenerateUniform, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Cfloat}, Csize_t),
                   generator, outputPtr, num)
end

@checked function curandGenerateUniformDouble(generator, outputPtr, num)
    initialize_api()
    ccall((:curandGenerateUniformDouble, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Cdouble}, Csize_t),
                   generator, outputPtr, num)
end

@checked function curandGenerateNormal(generator, outputPtr, n, mean, stddev)
    initialize_api()
    ccall((:curandGenerateNormal, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat),
                   generator, outputPtr, n, mean, stddev)
end

@checked function curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev)
    initialize_api()
    ccall((:curandGenerateNormalDouble, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble),
                   generator, outputPtr, n, mean, stddev)
end

@checked function curandGenerateLogNormal(generator, outputPtr, n, mean, stddev)
    initialize_api()
    ccall((:curandGenerateLogNormal, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat),
                   generator, outputPtr, n, mean, stddev)
end

@checked function curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev)
    initialize_api()
    ccall((:curandGenerateLogNormalDouble, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble),
                   generator, outputPtr, n, mean, stddev)
end

@checked function curandCreatePoissonDistribution(lambda, discrete_distribution)
    initialize_api()
    ccall((:curandCreatePoissonDistribution, libcurand()), curandStatus_t,
                   (Cdouble, Ptr{curandDiscreteDistribution_t}),
                   lambda, discrete_distribution)
end

@checked function curandDestroyDistribution(discrete_distribution)
    initialize_api()
    ccall((:curandDestroyDistribution, libcurand()), curandStatus_t,
                   (curandDiscreteDistribution_t,),
                   discrete_distribution)
end

@checked function curandGeneratePoisson(generator, outputPtr, n, lambda)
    initialize_api()
    ccall((:curandGeneratePoisson, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cdouble),
                   generator, outputPtr, n, lambda)
end

@checked function curandGeneratePoissonMethod(generator, outputPtr, n, lambda, method)
    initialize_api()
    ccall((:curandGeneratePoissonMethod, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cdouble, curandMethod_t),
                   generator, outputPtr, n, lambda, method)
end

@checked function curandGenerateBinomial(generator, outputPtr, num, n, p)
    initialize_api()
    ccall((:curandGenerateBinomial, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{UInt32}, Csize_t, UInt32, Cdouble),
                   generator, outputPtr, num, n, p)
end

@checked function curandGenerateBinomialMethod(generator, outputPtr, num, n, p, method)
    initialize_api()
    ccall((:curandGenerateBinomialMethod, libcurand()), curandStatus_t,
                   (curandGenerator_t, CuPtr{UInt32}, Csize_t, UInt32, Cdouble,
                    curandMethod_t),
                   generator, outputPtr, num, n, p, method)
end

@checked function curandGenerateSeeds(generator)
    initialize_api()
    ccall((:curandGenerateSeeds, libcurand()), curandStatus_t,
                   (curandGenerator_t,),
                   generator)
end

@checked function curandGetDirectionVectors32(vectors, set)
    initialize_api()
    ccall((:curandGetDirectionVectors32, libcurand()), curandStatus_t,
                   (Ptr{Ptr{curandDirectionVectors32_t}}, curandDirectionVectorSet_t),
                   vectors, set)
end

@checked function curandGetScrambleConstants32(constants)
    initialize_api()
    ccall((:curandGetScrambleConstants32, libcurand()), curandStatus_t,
                   (Ptr{Ptr{UInt32}},),
                   constants)
end

@checked function curandGetDirectionVectors64(vectors, set)
    initialize_api()
    ccall((:curandGetDirectionVectors64, libcurand()), curandStatus_t,
                   (Ptr{Ptr{curandDirectionVectors64_t}}, curandDirectionVectorSet_t),
                   vectors, set)
end

@checked function curandGetScrambleConstants64(constants)
    initialize_api()
    ccall((:curandGetScrambleConstants64, libcurand()), curandStatus_t,
                   (Ptr{Ptr{Culonglong}},),
                   constants)
end
