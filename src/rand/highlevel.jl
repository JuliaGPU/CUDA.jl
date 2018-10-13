const GLOBAL_RNG = Ref{RNG}()
function global_rng()
    if !isassigned(GLOBAL_RNG)
        GLOBAL_RNG[] = create_generator()
    end
    GLOBAL_RNG[]
end

"""Generate uniformly distributed numbers"""
curand(args...; kwargs...)  = rand(global_rng(), args...; kwargs...)

"""Generate normally distributed numbers"""
curandn(args...; kwargs...) = randn(global_rng(), args...; kwargs...)

"""Generate log-normally distributed numbers"""
curand_logn(args...; kwargs...) = rand_logn(global_rng(), args...; kwargs...)

"""Generate Poisson distributed numbers"""
curand_poisson(args...; kwargs...) = rand_poisson(global_rng(), args...; kwargs...)


# uniform
Random.rand(rng::RNG, ::Type{Float32}, dims::Dims) = generate_uniform(rng, prod(dims))
Random.rand(rng::RNG, ::Type{Float64}, dims::Dims) = generate_uniform_double(rng, prod(dims))


# normal
Random.randn(rng::RNG, ::Type{Float32}, dims::Dims; mean=0, stddev=1) =
    generate_normal(rng, prod(dims), mean, stddev)
Random.randn(rng::RNG, ::Type{Float64}, dims::Dims; mean=0, stddev=1) =
    generate_normal_double(rng, prod(dims), mean, stddev)
## Base interface
Random.randn(rng::RNG, T::Type, dims::Dims; kwargs...) = randn(rng, T, dims; kwargs...)
Random.randn(rng::RNG, T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    randn(rng, T, Dims((dim1, dims...)); kwargs...)

# log-normal
rand_logn(rng::RNG, ::Type{Float32}, dims::Dims, mean, stddev) =
    generate_log_normal(rng, prod(dims), mean, stddev)
rand_logn(rng::RNG, ::Type{Float64}, dims::Dims, mean, stddev) =
    generate_log_normal_double(rng, prod(dims), mean, stddev)
## Base-like typeless invocation
randn_logn(rng::RNG, dims::Dims, mean, stddev) =
    randn_logn(rng, Float64, n, mean, stddev)

# poisson
rand_poisson(rng::RNG, ::Type{Cuint}, dims::Dims; lambda=1) =
    generate_poisson(rng, prod(dims), lambda)
## Base-like typeless invocation
rand_poisson(rng::RNG, dims::Integer...; kwargs...) =
    rand_poisson(rng, Cuint, dims; kwargs...)
