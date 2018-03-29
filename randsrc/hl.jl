
const _rng = create_generator()
atexit(() -> destroy_generator(_rng))

# uniform
"""Generate n uniformly distributed numbers"""
curand(rng::RNG, ::Type{Float32}, n::Int) = generate_uniform(rng, UInt(n))
curand(::Type{Float32}, n::Int) = curand(_rng, Float64, n)

curand(rng::RNG, ::Type{Float64}, n::Int) = generate_uniform_double(rng, UInt(n))
curand(::Type{Float64}, n::Int) = curand(_rng, Float64, n)
curand(n::Int) = curand(Float64, n)


# normal
"""
Generate n normally distributed numbers with specified mean and
standard deviation
"""
curandn(rng::RNG, ::Type{Float32}, n::Int, mean::Float32, stddev::Float32) =
    generate_normal(rng, UInt(n), mean, stddev)
curandn(::Type{Float32}, n::Int, mean::Float32, stddev::Float32) =
    curandn(_rng, Float32, n, mean, stddev)

curandn(rng::RNG, ::Type{Float64}, n::Int, mean::Float64, stddev::Float64) =
    generate_normal_double(rng, UInt(n), mean, stddev)
curandn(::Type{Float64}, n::Int, mean::Float64, stddev::Float64) =
    curandn(_rng, Float64, n, mean, stddev)
curandn(n::Int, mean::Float64, stddev::Float64) =
    curandn(Float64, n, mean, stddev)

# log-normal
"""
Generate n log-normally distributed numbers with specified mean and
standard deviation
"""
curand_logn(rng::RNG, ::Type{Float32}, n::Int, mean::Float32, stddev::Float32) =
    generate_log_normal(rng, UInt(n), mean, stddev)
curand_logn(::Type{Float32}, n::Int, mean::Float32, stddev::Float32) =
    curand_logn(_rng, Float32, n, mean, stddev)

curand_logn(rng::RNG, ::Type{Float64}, n::Int, mean::Float64, stddev::Float64) =
    generate_log_normal_double(rng, UInt(n), mean, stddev)
curand_logn(::Type{Float64}, n::Int, mean::Float64, stddev::Float64) =
    curand_logn(_rng, Float64, n, mean, stddev)
curand_logn(n::Int, mean::Float64, stddev::Float64) =
    curand_logn(Float64, n, mean, stddev)

# poisson
"""Generate n numbers according to Poisson distribution"""
curand_poisson(rng::RNG, n::Int, lambda::Float64) =
    generate_poisson(rng, UInt(n), lambda)
curand_poisson(n::Int, lambda::Float64) = curand_poisson(_rng, n, lambda)
