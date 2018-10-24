const GLOBAL_RNG = Ref{RNG}()
function global_rng()
    # create the CURAND generator lazily, because it consumes quite a bit of GPU memory
    if !isassigned(GLOBAL_RNG)
        GLOBAL_RNG[] = create_generator()
    end
    GLOBAL_RNG[]
end

# the CURAND RNG can only handle Float32 and Float64 (it actually differs between calls,
# eg. Poisson is Cuint but CURAND-specific so no fallback needed)
const CURANDFloat = Union{Float32,Float64}
array_rng(A::CuArray{<:CURANDFloat}) = global_rng()
array_rng(A::CuArray) = GPUArrays.global_rng(A)


# seeding

seed!(rng::RNG=global_rng()) = generate_seeds(rng)


# in-place

"""Populate an array with uniformly distributed numbers"""
curand!(A::CuArray; kwargs...) = rand!(array_rng(A), A; kwargs...)

"""Populate an array with normally distributed numbers"""
curandn!(A::CuArray; kwargs...) = randn!(array_rng(A), A; kwargs...)

"""Populate an array with log-normally distributed numbers"""
curand_logn!(args...; kwargs...) = rand_logn!(global_rng(), args...; kwargs...)

"""Populate an array with Poisson distributed numbers"""
curand_poisson!(args...; kwargs...) = rand_poisson!(global_rng(), args...; kwargs...)

# high-performance alternatives for commonly-used functions
Random.rand!(A::CuArray{<:CURANDFloat}) = curand!(A)
Random.randn!(A::CuArray{<:CURANDFloat}) = curandn!(A)

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


# out of place

"""Generate an array with uniformly distributed numbers"""
curand(args...; kwargs...)  = rand(global_rng(), args...; kwargs...)

"""Generate an array with normally distributed numbers"""
curandn(args...; kwargs...) = randn(global_rng(), args...; kwargs...)

"""Generate an array with log-normally distributed numbers"""
curand_logn(args...; kwargs...) = rand_logn(global_rng(), args...; kwargs...)

"""Generate an array with Poisson distributed numbers"""
curand_poisson(args...; kwargs...) = rand_poisson(global_rng(), args...; kwargs...)

# uniform
Random.rand(rng::RNG, ::Type{X}, dims::Dims) where {X} = rand!(rng, CuArray{X}(dims))
## typeless (prefer Float32)
Random.rand(rng::RNG, dims::Integer...; kwargs...) = rand(rng, Float32, dims...; kwargs...)

# normal
Random.randn(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} = randn!(rng, CuArray{X}(dims); kwargs...)
## convenience
Random.randn(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    randn(rng, X, Dims((dim1, dims...)); kwargs...)
## typeless (prefer Float32)
Random.randn(rng::RNG, dims::Integer...; kwargs...) = randn(rng, Float32, dims...; kwargs...)

# log-normal
rand_logn(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} = rand_logn!(rng, CuArray{X}(dims); kwargs...)
## convenience
rand_logn(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_logn(rng, X, Dims((dim1, dims...)); kwargs...)
## typeless (prefer Float32)
rand_logn(rng::RNG, dims...; kwargs...) = rand_logn(rng, Float32, dims...; kwargs...)

# poisson
rand_poisson(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} = rand_poisson!(rng, CuArray{X}(dims); kwargs...)
## convenience
rand_poisson(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_poisson(rng, X, Dims((dim1, dims...)); kwargs...)
## typeless (prefer Cuint)
rand_poisson(rng::RNG, dims...; kwargs...) = rand_poisson(rng, Cuint, dims...; kwargs...)
