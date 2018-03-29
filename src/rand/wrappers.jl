function create_generator(rng_type::Int=CURAND_RNG_PSEUDO_DEFAULT)
    aptr = Ptr{Void}[0]
    @check ccall((:curandCreateGenerator, libcurand),
                curandStatus_t, (Ptr{Void}, Cint), aptr, rng_type)
    return RNG(aptr[1], rng_type)
end

function create_generator_host(rng_type::Int=CURAND_RNG_PSEUDO_DEFAULT)
    aptr = Ptr{Void}[0]
    @check ccall((:curandCreateGeneratorHost, libcurand),
                curandStatus_t, (Ptr{Void}, Cint), aptr, rng_type)
    return RNG(aptr[1], rng_type)
end

function destroy_generator(rng::RNG)
    @check ccall((:curandDestroyGenerator, libcurand),
                curandStatus_t, (Ptr{Void},), rng.ptr)
end

function get_version()
    ver = Ref{Cint}(0)
    @check ccall((:curandGetVersion, libcurand),
                 curandStatus_t, (Ref{Cint},), ver)
    return ver[]
end

# TODO: curandSetStream

function set_pseudo_random_generator_seed(rng::RNG, seed::Int64)
    @check ccall((:curandSetPseudoRandomGeneratorSeed, libcurand),
                 curandStatus_t, (Ptr{Void}, Clonglong), rng.ptr, seed)
end

function set_generator_offset(rng::RNG, offset::Int64)
    @check ccall((:curandSetGeneratorOffset, libcurand),
                 curandStatus_t, (Ptr{Void}, Clonglong), rng.ptr, offset)
end

function set_generator_ordering(rng::RNG, order::Int)
    @check ccall((:curandSetGeneratorOrdering, libcurand),
                 curandStatus_t, (Ptr{Void}, Cint), rng.ptr, order)
end

function set_quasi_random_generator_dimensions(rng::RNG, num_dimensions::UInt)
    @check ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand),
                 curandStatus_t, (Ptr{Void}, Cuint),
                 rng.ptr, num_dimensions)
end


"""
Generate 64-bit quasirandom numbers.
"""
function generate(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{UInt32}(sz)
    @check ccall((:curandGenerate, libcurand),
                 curandStatus_t, (Ptr{Void}, Ptr{UInt32}, Csize_t),
                 rng.ptr, arr, n)
    return arr
end


"""
Generate uniformly distributed floats.

Valid RNG types are:
 - CURAND_RNG_QUASI_SOBOL64
 - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
"""
function generate_long_long(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{UInt64}(sz)
    @check ccall((:curandGenerateLongLong, libcurand),
                 curandStatus_t, (Ptr{Void}, Ptr{Culonglong}, Csize_t),
                 rng.ptr, arr, n)
    return arr
end

# uniform
function generate_uniform(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{Float32}(sz)
    @check ccall((:curandGenerateUniform, libcurand),
                 curandStatus_t, (Ptr{Void}, Ptr{Float32}, Csize_t),
                 rng.ptr, arr, n)
    return arr
end

function generate_uniform_double(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{Float64}(sz)
    @check ccall((:curandGenerateUniformDouble, libcurand),
                 curandStatus_t, (Ptr{Void}, Ptr{Float64}, Csize_t),
                 rng.ptr, arr, n)
    return arr
end

# normal
function generate_normal(rng::RNG, n::UInt, mean::Float32, stddev::Float32)
    sz = Int(n)
    arr = CuArray{Float32}(sz)
    @check ccall((:curandGenerateNormal, libcurand),
                 curandStatus_t,
                 (Ptr{Void}, Ptr{Float32}, Csize_t, Cfloat, Cfloat),
                 rng.ptr, arr, n, mean, stddev)
    return arr
end

function generate_normal_double(rng::RNG, n::UInt, mean::Float64, stddev::Float64)
    sz = Int(n)
    arr = CuArray{Float64}(sz)
    @check ccall((:curandGenerateNormalDouble, libcurand),
                 curandStatus_t,
                 (Ptr{Void}, Ptr{Float64}, Csize_t, Cdouble, Cdouble),
                 rng.ptr, arr, n, mean, stddev)
    return arr
end


# lognormal
function generate_log_normal(rng::RNG, n::UInt, mean::Float32, stddev::Float32)
    sz = Int(n)
    arr = CuArray{Float32}(sz)
    @check ccall((:curandGenerateLogNormal, libcurand),
                 curandStatus_t,
                 (Ptr{Void}, Ptr{Float32}, Csize_t, Cfloat, Cfloat),
                 rng.ptr, arr, n, mean, stddev)
    return arr
end

function generate_log_normal_double(rng::RNG, n::UInt, mean::Float64, stddev::Float64)
    sz = Int(n)
    arr = CuArray{Float64}(sz)
    @check ccall((:curandGenerateLogNormalDouble, libcurand),
                 curandStatus_t,
                 (Ptr{Void}, Ptr{Float64}, Csize_t, Cdouble, Cdouble),
                 rng.ptr, arr, n, mean, stddev)
    return arr
end

# Poisson
"""Construct the histogram array for a Poisson distribution."""
function create_poisson_distribtion(lambda::Float64)
    aptr = Ptr{Void}[0]
    @check ccall((:curandCreatePoissonDistribution, libcurand),
                 curandStatus_t, (Cdouble, Ptr{Void}), lambda, aptr)
    return DiscreteDistribution(aptr[1])
end

"""Destroy the histogram array for a discrete distribution (e.g. Poisson)."""
function destroy_distribution(dd::DiscreteDistribution)
    @check ccall((:curandDestroyDistribution, libcurand),
                 curandStatus_t, (Ptr{Void},), dd.ptr)
end

"""Generate Poisson-distributed unsigned ints."""
function generate_poisson(rng::RNG, n::UInt, lambda::Float64)
    sz = Int(n)
    arr = CuArray{UInt32}(sz)
    @check ccall((:curandGeneratePoisson, libcurand),
                 curandStatus_t,
                 (Ptr{Void}, Ptr{Cuint}, Csize_t, Cdouble),
                 rng.ptr, arr, n, lambda)
    return arr
end

# seeds
"""Generate the starting state of the generator. """
function generate_seeds(rng::RNG)
    @check ccall((:curandGenerateSeeds, libcurand),
                 curandStatus_t, (Ptr{Void},), rng.ptr)
end


# TODO: implement curandGetDirectionVectors32
# TODO: implement curandGetScrambleConstants32
# TODO: implement curandGetDirectionVectors64
# TODO: implement curandGetScrambleConstants64
