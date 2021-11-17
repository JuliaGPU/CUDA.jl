# integration with LinearAlgebra.jl

CuMatOrAdj{T} = Union{CuMatrix, LinearAlgebra.Adjoint{T, <:CuMatrix{T}}, LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
CuOrAdj{T} = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}}, LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}


# matrix division

function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy(_A), copy(_B)
    A, ipiv = CUSOLVER.getrf!(A)
    return CUSOLVER.getrs!('N', A, ipiv, B)
end

# patch JuliaLang/julia#40899 to create a CuArray
# (see https://github.com/JuliaLang/julia/pull/41331#issuecomment-868374522)
if VERSION >= v"1.7-"
_zeros(::Type{T}, b::AbstractVector, n::Integer) where {T} = CUDA.zeros(T, max(length(b), n))
_zeros(::Type{T}, B::AbstractMatrix, n::Integer) where {T} = CUDA.zeros(T, max(size(B, 1), n), size(B, 2))
function Base.:\(F::Union{LinearAlgebra.LAPACKFactorizations{<:Any,<:CuArray},
                          Adjoint{<:Any,<:LinearAlgebra.LAPACKFactorizations{<:Any,<:CuArray}}},
                 B::AbstractVecOrMat)
    m, n = size(F)
    if m != size(B, 1)
        throw(DimensionMismatch("arguments must have the same number of rows"))
    end

    TFB = typeof(oneunit(eltype(B)) / oneunit(eltype(F)))
    FF = Factorization{TFB}(F)

    # For wide problem we (often) compute a minimum norm solution. The solution
    # is larger than the right hand side so we use size(F, 2).
    BB = _zeros(TFB, B, n)

    if n > size(B, 1)
        # Underdetermined
        copyto!(view(BB, 1:m, :), B)
    else
        copyto!(BB, B)
    end

    ldiv!(FF, BB)

    # For tall problems, we compute a least squares solution so only part
    # of the rhs should be returned from \ while ldiv! uses (and returns)
    # the complete rhs
    return LinearAlgebra._cut_B(BB, 1:n)
end
end


# qr

using LinearAlgebra: AbstractQ

# AbstractQ's `size` is the size of the full matrix,
# while `Matrix(Q)` only gives the compact Q.
# See JuliaLang/julia#26591 and JuliaGPU/CUDA.jl#969.
CuMatrix{T}(Q::AbstractQ{S}) where {T,S} = convert(CuArray, Matrix{T}(Q))
CuMatrix(Q::AbstractQ{T}) where {T} = CuMatrix{T}(Q)
CuArray{T}(Q::AbstractQ) where {T} = CuMatrix{T}(Q)
CuArray(Q::AbstractQ) = CuMatrix(Q)


# dot

function LinearAlgebra.dot(x::StridedCuArray{T1}, y::StridedCuArray{T2}) where {T1,T2}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))

    function kernel(x, y, res::AbstractArray{T}, shuffle) where {T}
        index = threadIdx().x
        threads = blockDim().x
        block_stride = (length(x)-1i32) ÷ gridDim().x + 1i32
        start = (blockIdx().x - 1i32) * block_stride + 1i32
        stop = blockIdx().x * block_stride

        local_val = zero(T)

        for i in start-1i32+index:threads:stop
            @inbounds local_val += x[i] * y[i]
        end

        val = reduce_block(+, local_val, zero(T), shuffle)
        if threadIdx().x == 1i32
            @inbounds @atomic res[] += val
        end

        return
    end

    dev = device()
    let T = promote_type(T1, T2)
        res = zeros(T, 1)

        # be conservative about using shuffle instructions
        shuffle = T <: Union{Bool,
                             UInt8, UInt16, UInt32, UInt64, UInt128,
                             Int8, Int16, Int32, Int64, Int128,
                             Float16, Float32, Float64,
                             ComplexF16, ComplexF32, ComplexF64}

        # how many threads do we want?
        # reduce_block(shuffle=true) requires the block to consist of full warps.
        wanted_threads = shuffle ? nextwarp(dev, n) : n
        function compute_threads(max_threads)
            if wanted_threads > max_threads
                shuffle ? prevwarp(dev, max_threads) : max_threads
            else
                wanted_threads
            end
        end

        # how many threads can we launch?
        kernel = @cuda launch=false kernel(x, y, res, Val(shuffle))
        compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
        config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
        threads = compute_threads(config.threads)
        blocks = min(config.blocks, cld(n, config.blocks))
        shmem = compute_shmem(threads)
        kernel(x, y, res, Val(shuffle); threads, blocks, shmem)

        @allowscalar res[]
    end
end
