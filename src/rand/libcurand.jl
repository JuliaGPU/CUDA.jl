# Julia wrapper for header: curand.h
# Automatically generated using Clang.jl


function curandCreateGenerator(generator, rng_type)
    @check @runtime_ccall((:curandCreateGenerator, libcurand), curandStatus_t,
                 (Ptr{curandGenerator_t}, curandRngType_t),
                 generator, rng_type)
end

function curandCreateGeneratorHost(generator, rng_type)
    @check @runtime_ccall((:curandCreateGeneratorHost, libcurand), curandStatus_t,
                 (Ptr{curandGenerator_t}, curandRngType_t),
                 generator, rng_type)
end

function curandDestroyGenerator(generator)
    @check @runtime_ccall((:curandDestroyGenerator, libcurand), curandStatus_t,
                 (curandGenerator_t,),
                 generator)
end

function curandGetVersion(version)
    @check @runtime_ccall((:curandGetVersion, libcurand), curandStatus_t,
                 (Ptr{Cint},),
                 version)
end

function curandGetProperty(type, value)
    @check @runtime_ccall((:curandGetProperty, libcurand), curandStatus_t,
                 (libraryPropertyType, Ptr{Cint}),
                 type, value)
end

function curandSetStream(generator, stream)
    @check @runtime_ccall((:curandSetStream, libcurand), curandStatus_t,
                 (curandGenerator_t, CUstream),
                 generator, stream)
end

function curandSetPseudoRandomGeneratorSeed(generator, seed)
    @check @runtime_ccall((:curandSetPseudoRandomGeneratorSeed, libcurand), curandStatus_t,
                 (curandGenerator_t, Culonglong),
                 generator, seed)
end

function curandSetGeneratorOffset(generator, offset)
    @check @runtime_ccall((:curandSetGeneratorOffset, libcurand), curandStatus_t,
                 (curandGenerator_t, Culonglong),
                 generator, offset)
end

function curandSetGeneratorOrdering(generator, order)
    @check @runtime_ccall((:curandSetGeneratorOrdering, libcurand), curandStatus_t,
                 (curandGenerator_t, curandOrdering_t),
                 generator, order)
end

function curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions)
    @check @runtime_ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand), curandStatus_t,
                 (curandGenerator_t, UInt32),
                 generator, num_dimensions)
end

function curandGenerate(generator, outputPtr, num)
    @check @runtime_ccall((:curandGenerate, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{UInt32}, Csize_t),
                 generator, outputPtr, num)
end

function curandGenerateLongLong(generator, outputPtr, num)
    @check @runtime_ccall((:curandGenerateLongLong, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Culonglong}, Csize_t),
                 generator, outputPtr, num)
end

function curandGenerateUniform(generator, outputPtr, num)
    @check @runtime_ccall((:curandGenerateUniform, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Cfloat}, Csize_t),
                 generator, outputPtr, num)
end

function curandGenerateUniformDouble(generator, outputPtr, num)
    @check @runtime_ccall((:curandGenerateUniformDouble, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Cdouble}, Csize_t),
                 generator, outputPtr, num)
end

function curandGenerateNormal(generator, outputPtr, n, mean, stddev)
    @check @runtime_ccall((:curandGenerateNormal, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat),
                 generator, outputPtr, n, mean, stddev)
end

function curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev)
    @check @runtime_ccall((:curandGenerateNormalDouble, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble),
                 generator, outputPtr, n, mean, stddev)
end

function curandGenerateLogNormal(generator, outputPtr, n, mean, stddev)
    @check @runtime_ccall((:curandGenerateLogNormal, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat),
                 generator, outputPtr, n, mean, stddev)
end

function curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev)
    @check @runtime_ccall((:curandGenerateLogNormalDouble, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble),
                 generator, outputPtr, n, mean, stddev)
end

function curandCreatePoissonDistribution(lambda, discrete_distribution)
    @check @runtime_ccall((:curandCreatePoissonDistribution, libcurand), curandStatus_t,
                 (Cdouble, Ptr{curandDiscreteDistribution_t}),
                 lambda, discrete_distribution)
end

function curandDestroyDistribution(discrete_distribution)
    @check @runtime_ccall((:curandDestroyDistribution, libcurand), curandStatus_t,
                 (curandDiscreteDistribution_t,),
                 discrete_distribution)
end

function curandGeneratePoisson(generator, outputPtr, n, lambda)
    @check @runtime_ccall((:curandGeneratePoisson, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cdouble),
                 generator, outputPtr, n, lambda)
end

function curandGeneratePoissonMethod(generator, outputPtr, n, lambda, method)
    @check @runtime_ccall((:curandGeneratePoissonMethod, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{UInt32}, Csize_t, Cdouble, curandMethod_t),
                 generator, outputPtr, n, lambda, method)
end

function curandGenerateBinomial(generator, outputPtr, num, n, p)
    @check @runtime_ccall((:curandGenerateBinomial, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{UInt32}, Csize_t, UInt32, Cdouble),
                 generator, outputPtr, num, n, p)
end

function curandGenerateBinomialMethod(generator, outputPtr, num, n, p, method)
    @check @runtime_ccall((:curandGenerateBinomialMethod, libcurand), curandStatus_t,
                 (curandGenerator_t, CuPtr{UInt32}, Csize_t, UInt32, Cdouble,
                  curandMethod_t),
                 generator, outputPtr, num, n, p, method)
end

function curandGenerateSeeds(generator)
    @check @runtime_ccall((:curandGenerateSeeds, libcurand), curandStatus_t,
                 (curandGenerator_t,),
                 generator)
end

function curandGetDirectionVectors32(vectors, set)
    @check @runtime_ccall((:curandGetDirectionVectors32, libcurand), curandStatus_t,
                 (Ptr{Ptr{curandDirectionVectors32_t}}, curandDirectionVectorSet_t),
                 vectors, set)
end

function curandGetScrambleConstants32(constants)
    @check @runtime_ccall((:curandGetScrambleConstants32, libcurand), curandStatus_t,
                 (Ptr{Ptr{UInt32}},),
                 constants)
end

function curandGetDirectionVectors64(vectors, set)
    @check @runtime_ccall((:curandGetDirectionVectors64, libcurand), curandStatus_t,
                 (Ptr{Ptr{curandDirectionVectors64_t}}, curandDirectionVectorSet_t),
                 vectors, set)
end

function curandGetScrambleConstants64(constants)
    @check @runtime_ccall((:curandGetScrambleConstants64, libcurand), curandStatus_t,
                 (Ptr{Ptr{Culonglong}},),
                 constants)
end
