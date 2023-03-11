# interfacing with LinearAlgebra standard library


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
            @inbounds CUDA.@atomic res[1i32] += val
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

function gemv_dispatch!(Y::CuVector, A, B, alpha::Number=true, beta::Number=false)
    mA, nA = size(A)

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

    tA, dA = if A isa Transpose
        'T', parent(A)
    elseif A isa Adjoint
        'C', parent(A)
    else
        'N', A
    end

    T = eltype(Y)
    if T <: CublasFloat && A isa StridedCuArray{T} && B isa StridedCuArray{T}
        gemv!(tA, alpha, dA, B, beta, Y)
    else
        gemm_dispatch!(Y, A, B, alpha, beta)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(Y::CuVector, A::StridedCuMatrix, B::StridedCuVector, a::$NT, b::$NT) =
            gemv_dispatch!(Y, A, B, a, b)
        LinearAlgebra.mul!(Y::CuVector, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::StridedCuVector, a::$NT, b::$NT) =
            gemv_dispatch!(Y, A, B, a, b)
        LinearAlgebra.mul!(Y::CuVector, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::StridedCuVector, a::$NT, b::$NT) =
            gemv_dispatch!(Y, A, B, a, b)
    end
end

# triangular

## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:DenseCuMatrix},
                            b::StridedCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trmv!($uploc, 'N', $isunitc, parent(A), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:DenseCuMatrix},
                            B::StridedCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trsv!($uploc, 'N', $isunitc, parent(A), B)
    end
end

## adjoint/transpose multiplication ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            b::DenseCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            b::DenseCuVector{T}) where {T<:CublasReal} =
            CUBLAS.trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            b::DenseCuVector{T}) where {T<:CublasComplex} =
            CUBLAS.trmv!($uploc, 'C', $isunitc, parent(parent(A)), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            B::StridedCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::StridedCuVector{T}) where {T<:CublasReal} =
            CUBLAS.trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::StridedCuVector{T}) where {T<:CublasComplex} =
            CUBLAS.trsv!($uploc, 'C', $isunitc, parent(parent(A)), B)
    end
end



#
# BLAS 3
#

# GEMM

function gemm_dispatch!(C::CuVecOrMat, A, B, alpha::Number=true, beta::Number=false)
    if ndims(A) > 2
        throw(ArgumentError("A has more than 2 dimensions"))
    elseif ndims(B) > 2
        throw(ArgumentError("B has more than 2 dimensions"))
    end
    mA, nA = size(A,1), size(A,2)
    mB, nB = size(B,1), size(B,2)

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

    tA, dA = if A isa Transpose
        'T', parent(A)
    elseif A isa Adjoint
        'C', parent(A)
    else
        'N', A
    end

    tB, dB = if B isa Transpose
        'T', parent(B)
    elseif B isa Adjoint
        'C', parent(B)
    else
        'N', B
    end

    T = eltype(C)
    if dA isa DenseCuArray && dB isa DenseCuArray &&
       gemmExComputeType(eltype(A), eltype(B), eltype(C), mA, nA, nB) !== nothing
        gemmEx!(tA, tB, alpha, dA, dB, beta, C)
    elseif T <: CublasFloat && dA isa DenseCuArray{T} && dB isa DenseCuArray{T}
        gemm!(tA, tB, alpha, dA, dB, beta, C)
    else
        GPUArrays.generic_matmatmul!(C, A, B, alpha, beta)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(C::CuMatrix, A::StridedCuVecOrMat, B::StridedCuVecOrMat, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::StridedCuMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::StridedCuMatrix, B::Transpose{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::Transpose{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::StridedCuMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::StridedCuMatrix, B::Adjoint{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::Adjoint{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::Adjoint{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::Transpose{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
    end
end

# triangular

## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:DenseCuMatrix},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, B)
        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{T,<:DenseCuMatrix}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, A)

        # optimization: Base.mul! uses lmul!/rmul! with a copy (because of BLAS)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{T,<:DenseCuMatrix},
                           B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{T,<:DenseCuMatrix}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, X)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:DenseCuMatrix},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trsm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B)

        # Right division
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{T,<:DenseCuMatrix}) where {T<:CublasFloat} =
            CUBLAS.trsm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A)

        # Matrix inverse
        function LinearAlgebra.inv(x::$t{T, <:CuMatrix{T}}) where T<:CublasFloat
            out = CuArray{T}(I(size(x,1)))
            $t(LinearAlgebra.ldiv!(x, out))
        end
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


## adjoint/transpose multiplication ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, B)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasComplex} =
            CUBLAS.trmm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B, B)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasReal} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, B)

        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, A)
        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasComplex} =
            CUBLAS.trmm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A, A)
        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasReal} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, A)

        # optimization: Base.mul! uses lmul!/rmul! with a copy (because of BLAS)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                           B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                           B::DenseCuMatrix{T}) where {T<:CublasComplex} =
            CUBLAS.trmm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                           B::DenseCuMatrix{T}) where {T<:CublasReal} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasComplex} =
            CUBLAS.trmm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasReal} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, X)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trsm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasReal} =
            CUBLAS.trsm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasComplex} =
            CUBLAS.trsm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B)

        # Right division
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}}) where {T<:CublasFloat} =
            CUBLAS.trsm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A)
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasReal} =
            CUBLAS.trsm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A)
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasComplex} =
            CUBLAS.trsm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A)
    end
end

function LinearAlgebra.mul!(X::DenseCuMatrix{T},
                            A::LowerTriangular{T,<:DenseCuMatrix},
                            B::UpperTriangular{T,<:DenseCuMatrix},
                            ) where {T<:CublasFloat}
    triu!(parent(B))
    trmm!('L', 'L', 'N', 'N', one(T), parent(A), parent(B), parent(X))
    X
end

function LinearAlgebra.mul!(X::DenseCuMatrix{T},
                            A::UpperTriangular{T,<:DenseCuMatrix},
                            B::LowerTriangular{T,<:DenseCuMatrix},
                            ) where {T<:CublasFloat}
    tril!(parent(B))
    trmm!('L', 'U', 'N', 'N', one(T), parent(A), parent(B), parent(X))
    X
end

for (trtype, valtype) in ((:Transpose, :CublasFloat),
                          (:Adjoint,   :CublasReal),
                          (:Adjoint,   :CublasComplex))
    @eval begin
        function LinearAlgebra.mul!(X::DenseCuMatrix{T},
                                    A::UpperTriangular{T,<:DenseCuMatrix},
                                    B::LowerTriangular{<:Any,<:$trtype{T,<:DenseCuMatrix}},
                                    ) where {T<:$valtype}
            # operation is reversed to avoid executing the tranpose
            triu!(parent(A))
            CUBLAS.trmm!('R', 'U', 'T', 'N', one(T), parent(parent(B)), parent(A), parent(X))
            X
        end

        function LinearAlgebra.mul!(X::DenseCuMatrix{T},
                                    A::UpperTriangular{<:Any,<:$trtype{T,<:DenseCuMatrix}},
                                    B::LowerTriangular{T,<:DenseCuMatrix},
                                    ) where {T<:$valtype}
            tril!(parent(B))
            CUBLAS.trmm!('L', 'L', 'T', 'N', one(T), parent(parent(A)), parent(B), parent(X))
            X
        end

        function LinearAlgebra.mul!(X::DenseCuMatrix{T},
                                    A::LowerTriangular{<:Any,<:$trtype{T,<:DenseCuMatrix}},
                                    B::UpperTriangular{T,<:DenseCuMatrix},
                                    ) where {T<:$valtype}
            triu!(parent(B))
            CUBLAS.trmm!('L', 'U', 'T', 'N', one(T), parent(parent(A)), parent(B), parent(X))
            X
        end

        function LinearAlgebra.mul!(X::DenseCuMatrix{T},
                                    A::LowerTriangular{T,<:DenseCuMatrix},
                                    B::UpperTriangular{<:Any,<:$trtype{T,<:DenseCuMatrix}},
                                    ) where {T<:$valtype}
            # operation is reversed to avoid executing the tranpose
            tril!(parent(A))
            CUBLAS.trmm!('R', 'L', 'T', 'N', one(T), parent(parent(B)), parent(A), parent(X))
            X
        end
    end
end

# symmetric mul!
# level 2
@inline function LinearAlgebra.mul!(y::CuVector{T}, A::Hermitian{T,<:CuMatrix}, x::CuVector{T},
             α::Number, β::Number) where {T<:CublasReal}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return CUBLAS.symv!(A.uplo, alpha, A.data, x, beta, y)
    else
        error("only supports BLAS type, got $T")
    end
end

@inline function LinearAlgebra.mul!(y::CuVector{T}, A::Hermitian{T,<:CuMatrix}, x::CuVector{T},
             α::Number, β::Number) where {T<:CublasComplex}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return CUBLAS.hemv!(A.uplo, alpha, A.data, x, beta, y)
    else
        error("only supports BLAS type, got $T")
    end
end

# level 3

@inline function LinearAlgebra.mul!(C::CuMatrix{T}, A::Hermitian{T,<:CuMatrix}, B::CuMatrix{T},
             α::Number, β::Number) where {T<:CublasReal}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return CUBLAS.symm!('L', A.uplo, alpha, A.data, B, beta, C)
    else
        error("only supports BLAS type, got $T")
    end
end
@inline function LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, B::Hermitian{T,<:CuMatrix},
             α::Number, β::Number) where {T<:CublasReal}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return CUBLAS.symm!('R', B.uplo, alpha, B.data, A, beta, C)
    else
        error("only supports BLAS type, got $T")
    end
end
@inline function LinearAlgebra.mul!(C::CuMatrix{T}, A::Hermitian{T,<:CuMatrix}, B::CuMatrix{T},
             α::Number, β::Number) where {T<:CublasComplex}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return CUBLAS.hemm!('L', A.uplo, alpha, A.data, B, beta, C)
    else
        error("only supports BLAS type, got $T")
    end
end
@inline function LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, B::Hermitian{T,<:CuMatrix},
             α::Number, β::Number) where {T<:CublasComplex}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return CUBLAS.hemm!('R', B.uplo, alpha, B.data, A, beta, C)
    else
        error("only supports BLAS type, got $T")
    end
end

op_wrappers = ((identity, T -> 'N', identity),
               (T -> :(Transpose{T, <:$T}), T -> 'T', A -> :(parent($A))),
               (T -> :(Adjoint{T, <:$T}), T -> T <: Real ? 'T' : 'C', A -> :(parent($A))))

for op in (:(+), :(-))
    for (wrapa, transa, unwrapa) in op_wrappers, (wrapb, transb, unwrapb) in op_wrappers
        TypeA = wrapa(:(CuMatrix{T}))
        TypeB = wrapb(:(CuMatrix{T}))
        @eval Base.$op(A::$TypeA, B::$TypeB) where {T <: CublasFloat} = CUBLAS.geam($transa(T), $transb(T), one(T), $(unwrapa(:A)), $(op)(one(T)), $(unwrapb(:B)))
    end
end
