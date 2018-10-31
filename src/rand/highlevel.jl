# high-level interface for CURAND
#
# the interface is split in two levels:
# - functions that extend the Random standard library, and take an RNG as first argument,
#   will only ever dispatch to CURAND and as a result are limited in the types they support.
# - functions that take an array will dispatch to either CURAND or GPUArrays
# - `cu`-prefixed functions are provided for constructing GPU arrays from only an eltype


## seeding

seed!(rng::RNG=generator()) = generate_seeds(rng)


## in-place

# uniform
Random.rand!(rng::RNG, A::CuArray{Float32}) = generate_uniform(rng, A)
Random.rand!(rng::RNG, A::CuArray{Float64}) = generate_uniform_double(rng, A)

# normal
Random.randn!(rng::RNG, A::CuArray{Float32}; mean=0, stddev=1) = generate_normal(rng, A, mean, stddev)
Random.randn!(rng::RNG, A::CuArray{Float64}; mean=0, stddev=1) = generate_normal_double(rng, A, mean, stddev)

# log-normal
rand_logn!(rng::RNG, A::CuArray{Float32}; mean=0, stddev=1) = generate_log_normal(rng, A, mean, stddev)
rand_logn!(rng::RNG, A::CuArray{Float64}; mean=0, stddev=1) = generate_log_normal_double(rng, A, mean, stddev)

# log-normal
rand_poisson!(rng::RNG, A::CuArray{Cuint}; lambda=1) = generate_poisson(rng, A, lambda)


## out of place

Random.rand(rng::RNG, ::Type{X}, dims::Dims) where {X} = rand!(rng, CuArray{X}(undef, dims))
Random.randn(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} = randn!(rng, CuArray{X}(undef, dims); kwargs...)
rand_logn(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} = rand_logn!(rng, CuArray{X}(undef, dims); kwargs...)
rand_poisson(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} = rand_poisson!(rng, CuArray{X}(undef, dims); kwargs...)

# specify default types
Random.rand(rng::RNG, dims::Integer...; kwargs...) = rand(rng, Float32, dims...; kwargs...)
Random.randn(rng::RNG, dims::Integer...; kwargs...) = randn(rng, Float32, dims...; kwargs...)
rand_logn(rng::RNG, dims...; kwargs...) = rand_logn(rng, Float32, dims...; kwargs...)
rand_poisson(rng::RNG, dims...; kwargs...) = rand_poisson(rng, Cuint, dims...; kwargs...)

# convenience
Random.randn(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    randn(rng, X, Dims((dim1, dims...)); kwargs...)
rand_logn(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_logn(rng, X, Dims((dim1, dims...)); kwargs...)
rand_poisson(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_poisson(rng, X, Dims((dim1, dims...)); kwargs...)


## functions that dispatch to either CURAND or GPUArrays

uniform_rng(::CuArray{<:Union{Float32,Float64}}) = generator()
uniform_rng(A::CuArray) = GPUArrays.global_rng(A)

normal_rng(::CuArray{<:Union{Float32,Float64}}) = generator()
normal_rng(::CuArray{T}) where {T} =
    error("CuArrays does not support generating normally distributed numbers of type $T")

logn_rng(::CuArray{<:Union{Float32,Float64}}) = generator()
logn_rng(::CuArray{T}) where {T} =
    error("CuArrays does not support generating lognormally distributed numbers of type $T")

poisson_rng(::CuArray{Cuint}) = generator()
poisson_rng(::CuArray{T}) where {T} =
    error("CuArrays does not support generating Poisson distributed numbers of type $T")


Random.rand!(A::CuArray; kwargs...) = rand!(uniform_rng(A), A; kwargs...)
Random.randn!(A::CuArray; kwargs...) = randn!(normal_rng(A), A; kwargs...)
rand_logn!(A::CuArray; kwargs...) = rand_logn!(logn_rng(A), A; kwargs...)
rand_poisson!(A::CuArray; kwargs...) = rand_poisson!(poisson_rng(A), A; kwargs...)


# need to prefix with `cu` to disambiguate from Random functions that return an Array
# TODO: `@gpu rand` with Cassette
curand(::Type{X}, args...; kwargs...) where {X} = rand!(CuArray{X}(undef, args...); kwargs...)
curandn(::Type{X}, args...; kwargs...) where {X} = randn!(CuArray{X}(undef, args...); kwargs...)
curand_logn(::Type{X}, args...; kwargs...) where {X} = rand_logn!(CuArray{X}(undef, args...); kwargs...)
curand_poisson(::Type{X}, args...; kwargs...) where {X} = rand_poisson!(CuArray{X}(undef, args...); kwargs...)

# specify default types
curand(args...; kwargs...) where {X} = curand(Float32, args...; kwargs...)
curandn(args...; kwargs...) where {X} = curandn(Float32, args...; kwargs...)
curand_logn(args...; kwargs...) where {X} = curand_logn(Float32, args...; kwargs...)
curand_poisson(args...; kwargs...) where {X} = curand_poisson(Cuint, args...; kwargs...)
