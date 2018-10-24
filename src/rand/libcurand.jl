function create_generator(typ::Int=CURAND_RNG_PSEUDO_DEFAULT)
    ptr = Ref{curandGenerator_t}()
    @check ccall((:curandCreateGenerator, libcurand),
                 curandStatus_t, (Ptr{curandGenerator_t}, Cint), ptr, typ)
    r = RNG(ptr[], typ)
    finalizer(destroy_generator, r)
    return r
end

function destroy_generator(rng::RNG)
    @check ccall((:curandDestroyGenerator, libcurand),
                 curandStatus_t, (curandGenerator_t,), rng.ptr)
end

function get_version()
    ver = Ref{Cint}()
    @check ccall((:curandGetVersion, libcurand),
                 curandStatus_t, (Ref{Cint},), ver)
    return ver[]
end

# TODO: curandSetStream

function set_pseudo_random_generator_seed(rng::RNG, seed::Int64)
    @check ccall((:curandSetPseudoRandomGeneratorSeed, libcurand),
                 curandStatus_t, (curandGenerator_t, Clonglong), rng, seed)
end

function set_generator_offset(rng::RNG, offset::Int64)
    @check ccall((:curandSetGeneratorOffset, libcurand),
                 curandStatus_t, (curandGenerator_t, Clonglong), rng, offset)
end

function set_generator_ordering(rng::RNG, order::Int)
    @check ccall((:curandSetGeneratorOrdering, libcurand),
                 curandStatus_t, (curandGenerator_t, Cint), rng, order)
end

function set_quasi_random_generator_dimensions(rng::RNG, num_dimensions::UInt)
    @check ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand),
                 curandStatus_t, (curandGenerator_t, Cuint),
                 rng, num_dimensions)
end


"""
Generate 64-bit quasirandom numbers.
"""
function generate(rng::RNG, arr::CuArray, n::UInt)
    @check ccall((:curandGenerate, libcurand),
                 curandStatus_t, (curandGenerator_t, Ptr{UInt32}, Csize_t),
                 rng, arr, length(arr))
    return arr
end


"""
Generate uniformly distributed floats.

Valid RNG types are:
 - CURAND_RNG_QUASI_SOBOL64
 - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
"""
function generate_long_long(rng::RNG, arr::CuArray)
    @check ccall((:curandGenerateLongLong, libcurand),
                 curandStatus_t, (curandGenerator_t, Ptr{Culonglong}, Csize_t),
                 rng, arr, length(arr))
    return arr
end

# uniform
function generate_uniform(rng::RNG, arr::CuArray)
    @check ccall((:curandGenerateUniform, libcurand),
                 curandStatus_t, (curandGenerator_t, Ptr{Float32}, Csize_t),
                 rng, arr, length(arr))
    return arr
end

function generate_uniform_double(rng::RNG, arr::CuArray)
    @check ccall((:curandGenerateUniformDouble, libcurand),
                 curandStatus_t, (curandGenerator_t, Ptr{Float64}, Csize_t),
                 rng, arr, length(arr))
    return arr
end

# normal
function generate_normal(rng::RNG, arr::CuArray, mean, stddev)
    @check ccall((:curandGenerateNormal, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Ptr{Cfloat}, Csize_t, Cfloat, Cfloat),
                 rng, arr, length(arr), mean, stddev)
    return arr
end

function generate_normal_double(rng::RNG, arr::CuArray, mean, stddev)
    @check ccall((:curandGenerateNormalDouble, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Ptr{Cdouble}, Csize_t, Cdouble, Cdouble),
                 rng, arr, length(arr), mean, stddev)
    return arr
end


# lognormal
function generate_log_normal(rng::RNG, arr::CuArray, mean, stddev)
    @check ccall((:curandGenerateLogNormal, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Ptr{Cfloat}, Csize_t, Cfloat, Cfloat),
                 rng, arr, length(arr), mean, stddev)
    return arr
end

function generate_log_normal_double(rng::RNG, arr::CuArray, mean, stddev)
    @check ccall((:curandGenerateLogNormalDouble, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Ptr{Cdouble}, Csize_t, Cdouble, Cdouble),
                 rng, arr, length(arr), mean, stddev)
    return arr
end

# Poisson
"""Construct the histogram array for a Poisson distribution."""
function create_poisson_distribtion(lambda)
    ptr = Ref{curandDiscreteDistribution_t}()
    @check ccall((:curandCreatePoissonDistribution, libcurand),
                 curandStatus_t,
                 (Cdouble, Ptr{curandDiscreteDistribution_t}),
                 lambda, ptr)
    dist = DiscreteDistribution(ptr[])
    finalizer(destroy_distribution, dist)
    return dist
end

"""Destroy the histogram array for a discrete distribution (e.g. Poisson)."""
function destroy_distribution(dist::DiscreteDistribution)
    @check ccall((:curandDestroyDistribution, libcurand),
                 curandStatus_t,
                 (curandDiscreteDistribution_t,),
                 dist)
end

"""Generate Poisson-distributed unsigned ints."""
function generate_poisson(rng::RNG, arr::CuArray, lambda)
    @check ccall((:curandGeneratePoisson, libcurand),
                 curandStatus_t,
                 (curandGenerator_t, Ptr{Cuint}, Csize_t, Cdouble),
                 rng, arr, length(arr), lambda)
    return arr
end

# seeds
"""Generate the starting state of the generator. """
function generate_seeds(rng::RNG)
    @check ccall((:curandGenerateSeeds, libcurand),
                 curandStatus_t, (curandGenerator_t,), rng)
end

# TODO: curandGetDirectionVectors32
# TODO: curandGetScrambleConstants32
# TODO: curandGetDirectionVectors64
# TODO: curandGetScrambleConstants64
