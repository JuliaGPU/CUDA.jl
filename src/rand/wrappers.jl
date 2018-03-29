
# handling status and errors
const MESSAGE_BY_STATUS =
    Dict(CURAND_STATUS_SUCCESS => "CURAND_STATUS_SUCCESS",
         CURAND_STATUS_VERSION_MISMATCH => "CURAND_STATUS_VERSION_MISMATCH",
         CURAND_STATUS_NOT_INITIALIZED => "CURAND_STATUS_NOT_INITIALIZED",
         CURAND_STATUS_ALLOCATION_FAILED => "CURAND_STATUS_ALLOCATION_FAILED",
         CURAND_STATUS_TYPE_ERROR => "CURAND_STATUS_TYPE_ERROR",
         CURAND_STATUS_OUT_OF_RANGE => "CURAND_STATUS_OUT_OF_RANGE",
         CURAND_STATUS_LENGTH_NOT_MULTIPLE => "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
         CURAND_STATUS_DOUBLE_PRECISION_REQUIRED =>
           "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
         CURAND_STATUS_LAUNCH_FAILURE => "CURAND_STATUS_LAUNCH_FAILURE",
         CURAND_STATUS_PREEXISTING_FAILURE => "CURAND_STATUS_PREEXISTING_FAILURE",
         CURAND_STATUS_INITIALIZATION_FAILED =>
           "CURAND_STATUS_INITIALIZATION_FAILED",
         CURAND_STATUS_ARCH_MISMATCH => "CURAND_STATUS_ARCH_MISMATCH",
         CURAND_STATUS_INTERNAL_ERROR => "CURAND_STATUS_INTERNAL_ERROR")

function error_message(status::UInt32)
    return "cuRAND operation failed with status: " * MESSAGE_BY_STATUS[status]
end


function statuscheck(status::UInt32)
    if status != CURAND_STATUS_SUCCESS
        warn(error_message(status) * " ($status)")
        Base.show_backtrace(STDOUT, backtrace())
        throw(error_message(status))
    end
end


function create_generator(rng_type::Int=CURAND_RNG_PSEUDO_DEFAULT)
    aptr = Ptr{Void}[0]
    statuscheck(ccall((:curandCreateGenerator, libcurand),
                      curandStatus_t, (Ptr{Void}, Cint), aptr, rng_type))
    return RNG(aptr[1], rng_type)
end

function create_generator_host(rng_type::Int=CURAND_RNG_PSEUDO_DEFAULT)
    aptr = Ptr{Void}[0]
    statuscheck(ccall((:curandCreateGeneratorHost, libcurand),
                      curandStatus_t, (Ptr{Void}, Cint), aptr, rng_type))
    return RNG(aptr[1], rng_type)
end

function destroy_generator(rng::RNG)
    statuscheck(ccall((:curandDestroyGenerator, libcurand),
                      curandStatus_t, (Ptr{Void},), rng.ptr))
end

function get_version()
    ver = Ref{Cint}(0)
    statuscheck(ccall((:curandGetVersion, libcurand),
                      curandStatus_t, (Ref{Cint},), ver))
    return ver[]
end

# TODO: curandSetStream

function set_pseudo_random_generator_seed(rng::RNG, seed::Int64)
    statuscheck(ccall((:curandSetPseudoRandomGeneratorSeed, libcurand),
                      curandStatus_t, (Ptr{Void}, Clonglong), rng.ptr, seed))
end

function set_generator_offset(rng::RNG, offset::Int64)
    statuscheck(ccall((:curandSetGeneratorOffset, libcurand),
                      curandStatus_t, (Ptr{Void}, Clonglong), rng.ptr, offset))
end

function set_generator_ordering(rng::RNG, order::Int)
    statuscheck(ccall((:curandSetGeneratorOrdering, libcurand),
                      curandStatus_t, (Ptr{Void}, Cint), rng.ptr, order))
end

function set_quasi_random_generator_dimensions(rng::RNG, num_dimensions::UInt)
    statuscheck(ccall((:curandSetQuasiRandomGeneratorDimensions, libcurand),
                      curandStatus_t, (Ptr{Void}, Cuint),
                      rng.ptr, num_dimensions))
end


"""
Generate 64-bit quasirandom numbers.
"""
function generate(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{UInt32}(sz)
    statuscheck(ccall((:curandGenerate, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{UInt32}, Csize_t),
                      rng.ptr, arr, n))
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
    statuscheck(ccall((:curandGenerateLongLong, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{Culonglong}, Csize_t),
                      rng.ptr, arr, n))
    return arr
end

# uniform
function generate_uniform(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{Float32}(sz)
    statuscheck(ccall((:curandGenerateUniform, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{Float32}, Csize_t),
                      rng.ptr, arr, n))
    return arr
end

function generate_uniform_double(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CuArray{Float64}(sz)
    statuscheck(ccall((:curandGenerateUniformDouble, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{Float64}, Csize_t),
                      rng.ptr, arr, n))
    return arr
end

# normal
function generate_normal(rng::RNG, n::UInt, mean::Float32, stddev::Float32)
    sz = Int(n)
    arr = CuArray{Float32}(sz)
    statuscheck(ccall((:curandGenerateNormal, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Float32}, Csize_t, Cfloat, Cfloat),
                      rng.ptr, arr, n, mean, stddev))
    return arr
end

function generate_normal_double(rng::RNG, n::UInt, mean::Float64, stddev::Float64)
    sz = Int(n)
    arr = CuArray{Float64}(sz)
    statuscheck(ccall((:curandGenerateNormalDouble, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Float64}, Csize_t, Cdouble, Cdouble),
                      rng.ptr, arr, n, mean, stddev))
    return arr
end


# lognormal
function generate_log_normal(rng::RNG, n::UInt, mean::Float32, stddev::Float32)
    sz = Int(n)
    arr = CuArray{Float32}(sz)
    statuscheck(ccall((:curandGenerateLogNormal, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Float32}, Csize_t, Cfloat, Cfloat),
                      rng.ptr, arr, n, mean, stddev))
    return arr
end

function generate_log_normal_double(rng::RNG, n::UInt, mean::Float64, stddev::Float64)
    sz = Int(n)
    arr = CuArray{Float64}(sz)
    statuscheck(ccall((:curandGenerateLogNormalDouble, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Float64}, Csize_t, Cdouble, Cdouble),
                      rng.ptr, arr, n, mean, stddev))
    return arr
end

# Poisson
"""Construct the histogram array for a Poisson distribution."""
function create_poisson_distribtion(lambda::Float64)
    aptr = Ptr{Void}[0]
    statuscheck(ccall((:curandCreatePoissonDistribution, libcurand),
                      curandStatus_t, (Cdouble, Ptr{Void}), lambda, aptr))
    return DiscreteDistribution(aptr[1])
end

"""Destroy the histogram array for a discrete distribution (e.g. Poisson)."""
function destroy_distribution(dd::DiscreteDistribution)
    statuscheck(ccall((:curandDestroyDistribution, libcurand),
                      curandStatus_t, (Ptr{Void},), dd.ptr))
end

"""Generate Poisson-distributed unsigned ints."""
function generate_poisson(rng::RNG, n::UInt, lambda::Float64)
    sz = Int(n)
    arr = CuArray{UInt32}(sz)
    statuscheck(ccall((:curandGeneratePoisson, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Cuint}, Csize_t, Cdouble),
                      rng.ptr, arr, n, lambda))
    return arr
end

# seeds
"""Generate the starting state of the generator. """
function generate_seeds(rng::RNG)
    statuscheck(ccall((:curandGenerateSeeds, libcurand),
                      curandStatus_t, (Ptr{Void},), rng.ptr))
end


# TODO: implement curandGetDirectionVectors32
# TODO: implement curandGetScrambleConstants32
# TODO: implement curandGetDirectionVectors64
# TODO: implement curandGetScrambleConstants64
