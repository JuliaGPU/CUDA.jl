function curandCreateGenerator(typ::Int=CURAND_RNG_PSEUDO_DEFAULT)
    ptr = Ref{curandGenerator_t}()
    @check ccall((:curandCreateGenerator, libcurand),
                 curandStatus_t,
                 (Ptr{curandGenerator_t}, Cint), ptr, typ)
    r = RNG(ptr[], typ)
    finalizer(curandDestroyGenerator, r)
    return r
end

function curandDestroyGenerator(rng::RNG)
    @check ccall((:curandDestroyGenerator, libcurand),
                 curandStatus_t,
                 (curandGenerator_t,), rng)
end

function curandGetVersion()
    ver = Ref{Cint}()
    @check ccall((:curandGetVersion, libcurand),
                 curandStatus_t,
                 (Ref{Cint},), ver)
    return ver[]
end

# TODO: curandSetStream

function curandSetPseudoRandomGeneratorSeed(rng::RNG, seed::Int64)
    @check ccall((:curandSetPseudoRandomGeneratorSeed, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Clonglong), rng, seed)
end

function curandSetGeneratorOffset(rng::RNG, offset::Int64)
    @check ccall((:curandSetGeneratorOffset, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Clonglong), rng, offset)
end

function curandSetGeneratorOrdering(rng::RNG, order::Int)
    @check ccall((:curandSetGeneratorOrdering, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Cint), rng, order)
end

function curandSetQuasiRandomGeneratorDimensions(rng::RNG, num_dimensions::UInt)
    @check ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Cuint),
                 rng, num_dimensions)
end


"""
Generate 64-bit quasirandom numbers.
"""
function curandGenerate(rng::RNG, arr::CuArray, n::UInt, num=length(arr))
    @check ccall((:curandGenerate, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{UInt32}, Csize_t),
                 rng, arr, num)
    return arr
end


"""
Generate uniformly distributed floats.

Valid RNG types are:
 - CURAND_RNG_QUASI_SOBOL64
 - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
"""
function curandGenerateLongLong(rng::RNG, arr::CuArray, num=length(arr))
    @check ccall((:curandGenerateLongLong, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Culonglong}, Csize_t),
                 rng, arr, num)
    return arr
end

# uniform
function curandGenerateUniform(rng::RNG, arr::CuArray, num=length(arr))
    @check ccall((:curandGenerateUniform, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Float32}, Csize_t),
                 rng, arr, num)
    return arr
end

function curandGenerateUniformDouble(rng::RNG, arr::CuArray, num=length(arr))
    @check ccall((:curandGenerateUniformDouble, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Float64}, Csize_t),
                 rng, arr, num)
    return arr
end

# normal
function curandGenerateNormal(rng::RNG, arr::CuArray, mean, stddev, num=length(arr))
    @check ccall((:curandGenerateNormal, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat),
                 rng, arr, num, mean, stddev)
    return arr
end

function curandGenerateNormalDouble(rng::RNG, arr::CuArray, mean, stddev, num=length(arr))
    @check ccall((:curandGenerateNormalDouble, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble),
                 rng, arr, num, mean, stddev)
    return arr
end


# lognormal
function curandGenerateLogNormal(rng::RNG, arr::CuArray, mean, stddev, num=length(arr))
    @check ccall((:curandGenerateLogNormal, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Cfloat}, Csize_t, Cfloat, Cfloat),
                 rng, arr, num, mean, stddev)
    return arr
end

function curandGenerateLogNormalDouble(rng::RNG, arr::CuArray, mean, stddev, num=length(arr))
    @check ccall((:curandGenerateLogNormalDouble, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Cdouble}, Csize_t, Cdouble, Cdouble),
                 rng, arr, num, mean, stddev)
    return arr
end

# Poisson
"""Construct the histogram array for a Poisson distribution."""
function curandCreatePoissonDistribution(lambda)
    ptr = Ref{curandDiscreteDistribution_t}()
    @check ccall((:curandCreatePoissonDistribution, libcurand),
                 curandStatus_t,
                 (Cdouble, Ptr{curandDiscreteDistribution_t}),
                 lambda, ptr)
    dist = DiscreteDistribution(ptr[])
    finalizer(curandDestroyDistribution, dist)
    return dist
end

"""Destroy the histogram array for a discrete distribution (e.g. Poisson)."""
function curandDestroyDistribution(dist::DiscreteDistribution)
    @check ccall((:curandDestroyDistribution, libcurand),
                 curandStatus_t,
                 (curandDiscreteDistribution_t,),
                 dist)
end

"""Generate Poisson-distributed unsigned ints."""
function curandGeneratePoisson(rng::RNG, arr::CuArray, lambda, num=length(arr))
    @check ccall((:curandGeneratePoisson, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, CuPtr{Cuint}, Csize_t, Cdouble),
                 rng, arr, num, lambda)
    return arr
end

# seeds
"""Generate the starting state of the generator. """
function curandGenerateSeeds(rng::RNG)
    @check ccall((:curandGenerateSeeds, libcurand),
                 curandStatus_t,
                 (curandGenerator_t,), rng)
end

# TODO: curandGetDirectionVectors32
# TODO: curandGetScrambleConstants32
# TODO: curandGetDirectionVectors64
# TODO: curandGetScrambleConstants64

function curandGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  @check ccall((:curandGetProperty, libcurand),
               curandStatus_t,
               (Cint, Ptr{Cint}),
               property, value_ref)
  value_ref[]
end
