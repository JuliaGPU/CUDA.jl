# interfacing with LinearAlgebra standard library

using LinearAlgebra: MulAddMul, AdjOrTrans, wrap, UpperOrLowerTriangular

#
# BLAS 1
#

LinearAlgebra.rmul!(x::StridedCuArray{<:CublasFloat}, k::Number) =
  scal!(length(x), k, x)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::DenseCuArray{<:CublasFloat}, k::Real) =
  invoke(rmul!, Tuple{typeof(x), Number}, x, k)

function LinearAlgebra.dot(x::StridedCuArray{T}, y::StridedCuArray{T}) where T<:Union{Float16, CublasReal}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    dot(n, x, y)
end

function LinearAlgebra.dot(x::StridedCuArray{T}, y::StridedCuArray{T}) where T<:Union{ComplexF16, CublasComplex}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    dotc(n, x, y)
end

# generic fallback
function LinearAlgebra.dot(x::AnyCuArray{T1}, y::AnyCuArray{T2}) where {T1,T2}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))

    # custom kernel using simple linear indexing and atomic additions,
    # resulting in about 10% speed-up compared to a simple mapreduce.
    function kernel(x, y, res::AbstractArray{T}, shuffle) where {T}
        local_val = zero(T)

        # grid-stride loop
        i = threadIdx().x + (blockIdx().x - 1i32)*blockDim().x
        while i <= length(x)
            @inbounds local_val += LinearAlgebra.dot(x[i], y[i])
            i += blockDim().x * gridDim().x
        end

        val = CUDA.reduce_block(+, local_val, zero(T), shuffle)
        if threadIdx().x == 1i32
            # NOTE: introduces nondeterminism
            @inbounds CUDA.@atomic res[] += val
        end

        return
    end

    dev = device()
    let T = promote_type(T1, T2)
        # only use the above kernel if we don't care about determinism
        # and if atomic operations are supported on these inputs
        atomic = if capability(device()) >= v"7.0"
            T <: Union{Int16, Int32, Int64, Float16, Float32, Float64}
        else
            T <: Union{Int32, Int64, Float32, Float64}
        end
        if CUDA.math_mode() == CUDA.PEDANTIC_MATH || !atomic
            return mapreduce((x,y)->LinearAlgebra.dot(x, y), +, x, y; init=zero(T))
        end

        res = CUDA.zeros(T, 1)

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

        CUDA.@allowscalar res[]
    end
end

function LinearAlgebra.:(*)(transx::Transpose{<:Any,<:StridedCuVector{T}}, y::StridedCuVector{T}) where T<:Union{ComplexF16, CublasComplex}
    x = transx.parent
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    return dotu(n, x, y)
end

function LinearAlgebra.norm(x::DenseCuArray{<:Union{Float16, ComplexF16, CublasFloat}}, p::Real=2)
    if p == 2
        return nrm2(x)
    else
        return invoke(norm, Tuple{AbstractGPUArray, Real}, x, p)
    end
end

LinearAlgebra.BLAS.asum(x::StridedCuArray{<:CublasFloat}) = asum(length(x), x)

function LinearAlgebra.axpy!(alpha::Number, x::StridedCuArray{T}, y::StridedCuArray{T}) where T<:Union{Float16, ComplexF16, CublasFloat}
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    axpy!(length(x), alpha, x, y)
end

function LinearAlgebra.axpby!(alpha::Number, x::StridedCuArray{T}, beta::Number, y::StridedCuArray{T}) where T<:Union{Float16, ComplexF16, CublasFloat}
    length(x)==length(y) || throw(DimensionMismatch("axpby arguments have lengths $(length(x)) and $(length(y))"))
    axpby!(length(x), alpha, x, beta, y)
end

function LinearAlgebra.rotate!(x::StridedCuArray{T}, y::StridedCuArray{T}, c::Number, s::Number) where T<:CublasFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("rotate arguments have lengths $nx and $ny"))
    rot!(nx, x, y, c, s)
end

function LinearAlgebra.reflect!(x::StridedCuArray{T}, y::StridedCuArray{T}, c::Number, s::Number) where T<:CublasFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("reflect arguments have lengths $nx and $ny"))
    rot!(nx, x, y, c, s)
    scal!(ny, -real(one(T)), y)
    x, y
end



#
# BLAS 2
#

# GEMV
# legacy method
LinearAlgebra.generic_matvecmul!(Y::StridedCuVector, tA::AbstractChar, A::StridedCuMatrix, B::StridedCuVector, _add::MulAddMul) =
    LinearAlgebra.generic_matvecmul!(Y, tA, A, B, _add.alpha, _add.beta)
function LinearAlgebra.generic_matvecmul!(Y::StridedCuVector, tA::AbstractChar, A::StridedCuMatrix, B::StridedCuVector, alpha::Number, beta::Number)
    mA, nA = tA == 'N' ? size(A) : reverse(size(A))

    if nA != length(B)
        throw(DimensionMismatch("second dimension of A, $nA, does not match length of B, $(length(B))"))
    end

    if mA != length(Y)
        throw(DimensionMismatch("first dimension of A, $mA, does not match length of Y, $(length(Y))"))
    end

    if mA == 0
        return Y
    end

    if nA == 0
        return rmul!(Y, 0)
    end

    T = eltype(Y)
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        if T <: CublasFloat && eltype(A) == eltype(B) == T
            if tA in ('N', 'T', 'C')
                return gemv!(tA, alpha, A, B, beta, Y)
            elseif tA in ('S', 's')
                return symv!(tA == 'S' ? 'U' : 'L', alpha, A, B, beta, Y)
            elseif tA in ('H', 'h')
                return hemv!(tA == 'H' ? 'U' : 'L', alpha, A, B, beta, Y)
            end
        end
    end
    LinearAlgebra.generic_matmatmul!(Y, tA, 'N', A, B, alpha, beta)
end

# triangular

## multiplication
LinearAlgebra.generic_trimatmul!(c::StridedCuVector{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, b::StridedCuVector{T}) where {T<:CublasFloat} =
    trmv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, A, c === b ? c : copyto!(c, b))

## division
LinearAlgebra.generic_trimatdiv!(C::StridedCuVector{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, B::StridedCuVector{T}) where {T<:CublasFloat} =
    trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, A, C === B ? C : copyto!(C, B))


#
# BLAS 3
#

# GEMM
LinearAlgebra.generic_matmatmul!(C::StridedCuVecOrMat, tA, tB, A::StridedCuVecOrMat, B::StridedCuVecOrMat, _add::MulAddMul) =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
function LinearAlgebra.generic_matmatmul!(C::StridedCuVecOrMat, tA, tB, A::StridedCuVecOrMat, B::StridedCuVecOrMat, alpha::Number, beta::Number)
    T = eltype(C)
    mA, nA = size(A, tA == 'N' ? 1 : 2), size(A, tA == 'N' ? 2 : 1)
    mB, nB = size(B, tB == 'N' ? 1 : 2), size(B, tB == 'N' ? 2 : 1)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 0 || nA == 0 || nB == 0
        if size(C) != (mA, nB)
            throw(DimensionMismatch("C has dimensions $(size(C)), should have ($mA,$nB)"))
        end
        return LinearAlgebra.rmul!(C, 0)
    end

    if all(in(('N', 'T', 'C')), (tA, tB))
        if A isa StridedCuArray && B isa StridedCuArray &&
        gemmExComputeType(eltype(A), eltype(B), eltype(C), mA, nA, nB) !== nothing
            return gemmEx!(tA, tB, alpha, A, B, beta, C)
        elseif T <: CublasFloat && A isa StridedCuArray{T} && B isa StridedCuArray{T}
            return gemm!(tA, tB, alpha, A, B, beta, C)
        end
    end
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        # TODO: should the gemm part above be included in this branch?
        if (tA == 'S' || tA == 's') && tB == 'N'
            return symm!('L', tA == 'S' ? 'U' : 'L', alpha, A, B, beta, C)
        elseif (tB == 'S' || tB == 's') && tA == 'N'
            return symm!('R', tB == 'S' ? 'U' : 'L', alpha, B, A, beta, C)
        elseif (tA == 'H' || tA == 'h') && tB == 'N'
            return hemm!('L', tA == 'H' ? 'U' : 'L', alpha, A, B, beta, C)
        elseif (tB == 'H' || tB == 'h') && tA == 'N'
            return hemm!('R', tB == 'H' ? 'U' : 'L', alpha, B, A, beta, C)
        end
    end

    GPUArrays.generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), alpha, beta)
end


# triangular

LinearAlgebra.generic_trimatmul!(C::StridedCuMatrix{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trmm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
LinearAlgebra.generic_mattrimul!(C::StridedCuMatrix{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trmm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, A, C)

## tri-tri-mul!
const AdjOrTransOrCuMatrix{T} = Union{StridedCuMatrix{T}, AdjOrTrans{<:T,<:StridedCuMatrix}}
function LinearAlgebra.generic_trimatmul!(C::StridedCuMatrix{T}, uplocA, isunitcA, tfunA::Function, A::StridedCuMatrix{T}, triB::UpperOrLowerTriangular{T,<:AdjOrTransOrCuMatrix{T}}) where {T<:CublasFloat}
    uplocB = LinearAlgebra.uplo_char(triB)
    isunitcB = LinearAlgebra.isunit_char(triB)
    B = parent(triB)
    tfunB = LinearAlgebra.wrapperop(B)
    transa = tfunA === identity ? 'N' : tfunA === transpose ? 'T' : 'C'
    transb = tfunB === identity ? 'N' : tfunB === transpose ? 'T' : 'C'
    if uplocA == 'L' && tfunA === identity && tfunB === identity && uplocB == 'U' && isunitcB == 'N' # lower * upper
        triu!(B)
        trmm!('L', uplocA, transa, isunitcA, one(T), A, B, C)
    elseif uplocA == 'U' && tfunA === identity && tfunB === identity && uplocB == 'L' && isunitcB == 'N' # upper * lower
        tril!(B)
        trmm!('L', uplocA, transa, isunitcA, one(T), A, B, C)
    elseif uplocA == 'U' && tfunA === identity && tfunB !== identity && uplocB == 'U' && isunitcA == 'N'
        # operation is reversed to avoid executing the tranpose
        triu!(A)
        trmm!('R', uplocB, transb, isunitcB, one(T), parent(B), A, C)
    elseif uplocA == 'L' && tfunA !== identity && tfunB === identity && uplocB == 'L' && isunitcB == 'N'
        tril!(B)
        trmm!('L', uplocA, transa, isunitcA, one(T), A, B, C)
    elseif uplocA == 'U' && tfunA !== identity && tfunB === identity && uplocB == 'U' && isunitcB == 'N'
        triu!(B)
        trmm!('L', uplocA, transa, isunitcA, one(T), A, B, C)
    elseif uplocA == 'L' && tfunA === identity && tfunB !== identity && uplocB == 'L' && isunitcA == 'N'
        tril!(A)
        trmm!('R', uplocB, transb, isunitcB, one(T), parent(B), A, C)
    else
        throw("mixed triangular-triangular multiplication") # TODO: rethink
    end
    return C
end

LinearAlgebra.generic_trimatdiv!(C::StridedCuMatrix{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, B::AbstractMatrix{T}) where {T<:CublasFloat} =
    trsm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, C === B ? C : copyto!(C, B))
LinearAlgebra.generic_mattridiv!(C::StridedCuMatrix{T}, uploc, isunitc, tfun::Function, A::AbstractMatrix{T}, B::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trsm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, C === A ? C : copyto!(C, A))

# Matrix inverse
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
    (:UnitLowerTriangular, 'L', 'U'),
    (:UpperTriangular, 'U', 'N'),
    (:UnitUpperTriangular, 'U', 'U'))
    @eval function LinearAlgebra.inv(x::$t{T, <:CuMatrix{T}}) where T<:CublasFloat
        out = CuArray{T}(I(size(x,1)))
        $t(LinearAlgebra.ldiv!(x, out))
    end
end

# Diagonal
Base.Array(D::Diagonal{T, <:CuArray{T}}) where {T} = Diagonal(Array(D.diag))
CuArray(D::Diagonal{T, <:Vector{T}}) where {T} = Diagonal(CuArray(D.diag))

function LinearAlgebra.inv(D::Diagonal{T, <:CuArray{T}}) where {T}
    Di = map(inv, D.diag)
    if any(isinf, Di)
        error("Singular Exception")
    end
    Diagonal(Di)
end

LinearAlgebra.rdiv!(A::CuArray, D::Diagonal) =  _rdiv!(A, A, D)

Base.:/(A::CuArray, D::Diagonal) = _rdiv!(similar(A, typeof(oneunit(eltype(A)) / oneunit(eltype(D)))), A, D)

function _rdiv!(B::CuArray, A::CuArray, D::Diagonal)
    m, n = size(A, 1), size(A, 2)
    if (k = length(D.diag)) != n
        throw(DimensionMismatch("left hand side has $n columns but D is $k by $k"))
    end
    B .= A*inv(D)
    B
end

# diagonal mul!
function LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, B::Diagonal{T,<:CuVector}) where {T<:CublasFloat}
    return dgmm!('R', A, B.diag, C)
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Diagonal{T,<:CuVector}, B::CuMatrix{T}) where {T<:CublasFloat}
    return dgmm!('L', B, A.diag, C)
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Transpose{T,<:CuMatrix}, B::Diagonal{T,<:CuVector}) where {T<:CublasFloat}
    C .= A
    C .*= B.diag'
    return C
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Diagonal{T,<:CuVector}, B::Transpose{T,<:CuMatrix}) where {T<:CublasFloat}
    C .= B
    C .*= A.diag
    return C
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Adjoint{T,<:CuMatrix}, B::Diagonal{T,<:CuVector}) where {T<:CublasFloat}
    C .= A
    C .*= B.diag'
    return C
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Diagonal{T,<:CuVector}, B::Adjoint{T,<:CuMatrix}) where {T<:CublasFloat}
    C .= B
    C .*= A.diag
    return C
end

# symmetric mul!

op_wrappers = ((identity, T -> 'N', identity),
               (T -> :(Transpose{T, <:$T}), T -> 'T', A -> :(parent($A))),
               (T -> :(Adjoint{T, <:$T}), T -> T <: Real ? 'T' : 'C', A -> :(parent($A))))

for op in (:(+), :(-))
    for (wrapa, transa, unwrapa) in op_wrappers, (wrapb, transb, unwrapb) in op_wrappers
        TypeA = wrapa(:(CuMatrix{T}))
        TypeB = wrapb(:(CuMatrix{T}))
        @eval Base.$op(A::$TypeA, B::$TypeB) where {T <: CublasFloat} = geam($transa(T), $transb(T), one(T), $(unwrapa(:A)), $(op)(one(T)), $(unwrapb(:B)))
    end
end

# Kronecker product
function LinearAlgebra.kron!(C::CuMatrix{TC}, A::CuMatrix{TA}, B::CuMatrix{TB}) where {TA,TB,TC}

    function _kron_mat_kernelA!(C, A, B, m, n, p, q)
        index_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        index_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

        stride_i = blockDim().x * gridDim().x
        stride_j = blockDim().y * gridDim().y

        index_i > m && return
        index_j > n && return

        for i in index_i:stride_i:m
            for j in index_j:stride_j:n
                for k in 1:p
                    for l in 1:q
                        @inbounds C[(i-1)*p+k, (j-1)*q+l] = A[i,j] * B[k,l]
                    end
                end
            end
        end
        return nothing
    end

    function _kron_mat_kernelB!(C, A, B, m, n, p, q)
        index_p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        index_q = (blockIdx().y - 1) * blockDim().y + threadIdx().y

        stride_p = blockDim().x * gridDim().x
        stride_q = blockDim().y * gridDim().y

        index_p > p && return
        index_q > q && return

        for i in 1:m
            for j in 1:n
                for k in index_p:stride_p:p
                    for l in index_q:stride_q:q
                        @inbounds C[(i-1)*p+k, (j-1)*q+l] = A[i,j] * B[k,l]
                    end
                end
            end
        end
        return nothing
    end

    m, n = size(A)
    p, q = size(B)

    # Use different kernels depending on the size of the matrices
    # choosing to parallelize the matrix with the largest number of elements
    m*n >= p*q ? (kernel = @cuda launch=false _kron_mat_kernelA!(C, A, B, m, n, p, q)) :
                 (kernel = @cuda launch=false _kron_mat_kernelB!(C, A, B, m, n, p, q))

    m*n >= p*q ? (sizes = (m, n)) : (sizes = (p, q))

    config = launch_configuration(kernel.fun)
    dim_ratio = sizes[1] / sizes[2]
    max_threads_i = max(1, floor(Int, sqrt(config.threads * dim_ratio)))
    max_threads_j = max(1, floor(Int, sqrt(config.threads / dim_ratio)))
    max_blocks_i = max(1, floor(Int, sqrt(config.blocks * dim_ratio)))
    max_blocks_j = max(1, floor(Int, sqrt(config.blocks / dim_ratio)))

    threads_i = min(sizes[1], max_threads_i)
    threads_j = min(sizes[2], max_threads_j)
    threads = (threads_i, threads_j)
    blocks_i = min(cld(sizes[1], threads_i), max_blocks_i)
    blocks_j = min(cld(sizes[2], threads_j), max_blocks_j)
    blocks = (blocks_i, blocks_j)

    kernel(C, A, B, m, n, p, q; threads=threads, blocks=blocks)

    return C
end

function LinearAlgebra.kron(A::CuMatrix{TA}, B::CuMatrix{TB}) where {TA,TB}
    m, n = size(A)
    p, q = size(B)

    T = promote_type(TA, TB)
    C = similar(A, T, m*p, n*q)

    kron!(C, A, B)
end
