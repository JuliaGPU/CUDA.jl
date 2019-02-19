
# Implements the Hillis--Steele algorithm using global memory
# See algorithm 1 at https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithm

function accumulate!(op::Function, vout::CuVector{T}, v::CuVector) where T

    vin = convert.(T, v)  # convert to type T

    Δ = 1   # Δ = 2^d
    n = ceil(Int, log2(length(v)))

    num_threads = 256
    num_blocks = ceil(Int, length(v) / num_threads)

    for d in 0:n   # passes through data

        @cuda blocks=num_blocks threads=num_threads _partial_accumulate!(op, vout, vin, Δ)

        vin, vout = vout, vin
        Δ *= 2
    end

    return vin
end


function _partial_accumulate!(op, vout, vin, Δ)
    @inbounds begin

        k = threadIdx().x + (blockIdx().x-1)*blockDim().x

        if k <= length(vin)
            if k > Δ
                vout[k] = vin[k - Δ] + vin[k]
            else
                vout[k] = vin[k]
            end
        end
    end

    return nothing

end
