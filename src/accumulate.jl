# Implements the Hillis--Steele algorithm using global memory
# See algorithm 1 at https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithm

# TODO: features
# - init::Some
# - CuMatrix
# - pairwise

# TODO: performance
# - shared memory / shuffle (see CUDAnative.jl/examples/scan)

function Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, dims::Int,
                           init::Nothing) where {T}
    if dims != 1
        return copyto!(vout, v)
    end

    return Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, nothing, nothing)
end

function Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, dims::Nothing,
                           init::Nothing) where {T}
    vin = T.(v)  # convert to vector with eltype T

    Δ = 1   # Δ = 2^d
    n = ceil(Int, log2(length(v)))

    # partial in-place accumulation
    function kernel(op, vout, vin, Δ)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if i <= length(vin)
            if i > Δ
                vout[i] = op(vin[i - Δ], vin[i])
            else
                vout[i] = vin[i]
            end
        end

        return
    end

    function configurator(kernel)
        fun = kernel.fun
        config = launch_configuration(fun)
        blocks = cld(length(v), config.threads)

        return (threads=config.threads, blocks=blocks)
    end

    for d in 0:n   # passes through data
        @cuda config=configurator kernel(op, vout, vin, Δ)

        vin, vout = vout, vin
        Δ *= 2
    end

    return vin
end

Base.accumulate_pairwise!(op, result::CuVector, v::CuVector) = accumulate!(op, result, v)
