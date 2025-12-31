# interfacing with LinearAlgebra standard library

using LinearAlgebra: MulAddMul, AdjOrTrans, wrap, UpperOrLowerTriangular, rmul!, lmul!
@static if VERSION ≥ v"1.12.0-rc"
    # we need to use the generic wrapper to avoid dispatch to the 2x2or3x3 method
    using LinearAlgebra: generic_matmatmul_wrapper!, BlasFlag, _symm_hemm_generic!
end
#
# BLAS 1
#

function LinearAlgebra.rmul!(x::CuArray{<:CublasFloat}, k::Bool)
    # explicitly fill x with zero to comply with julias "false = strong zero"
    !k && fill!(x, zero(eltype(x)))
    return x
end

LinearAlgebra.rmul!(x::StridedCuArray{<:CublasFloat}, k::Number) =
  scal!(length(x), k, x)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::DenseCuArray{<:CublasFloat}, k::Real) =
  invoke(rmul!, Tuple{typeof(x), Number}, x, k)

function LinearAlgebra.dot(x::StridedCuVector{T},
                           y::StridedCuVector{T}) where T<:Union{Float16, CublasReal}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    dot(n, x, y)
end

function LinearAlgebra.dot(x::StridedCuVector{T},
                           y::StridedCuVector{T}) where T<:Union{ComplexF16, CublasComplex}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    dotc(n, x, y)
end

# resolve ambiguities with generic wrapper below
LinearAlgebra.dot(x::CuArray{T}, y::CuArray{T}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64} =
    invoke(LinearAlgebra.dot, Tuple{StridedCuArray{T}, StridedCuArray{T}}, x, y)

# generic fallback
function LinearAlgebra.dot(x::AnyCuArray{T1}, y::AnyCuArray{T2}) where {T1,T2}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))

    n == 0 && return zero(promote_type(T1, T2))
    # custom kernel using simple linear indexing and atomic additions,
    # resulting in about 10% speed-up compared to a simple mapreduce.
    # COV_EXCL_START
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
    # COV_EXCL_STOP

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

# three-argument dot: avoid materializing A*y
function LinearAlgebra.dot(x::AnyCuArray{T1}, A::AnyCuArray{T2}, y::AnyCuArray{T3}) where {T1,T2,T3}
    nx = length(x)
    ny = length(y)
    mA, nA = size(A)

    nx == mA || throw(DimensionMismatch("length of x, $nx, does not match first dimension of A, $mA"))
    ny == nA || throw(DimensionMismatch("length of y, $ny, does not match second dimension of A, $nA"))

    # custom kernel using simple linear indexing and atomic additions
    # COV_EXCL_START
    function kernel(x, A, y, res::AbstractArray{T}, shuffle) where {T}
        local_val = zero(T)

        # grid-stride loop over rows
        i = threadIdx().x + (blockIdx().x - 1i32)*blockDim().x
        while i <= mA
            row_val = zero(T)
            for j in 1:nA
                # XXX: this is slow, but the focus is on avoiding materializing A*y
                @inbounds row_val += A[i, j] * y[j]
            end
            @inbounds local_val += LinearAlgebra.dot(x[i], row_val)
            i += blockDim().x * gridDim().x
        end

        val = CUDA.reduce_block(+, local_val, zero(T), shuffle)
        if threadIdx().x == 1i32
            # NOTE: introduces nondeterminism
            @inbounds CUDA.@atomic res[] += val
        end

        return
    end
    # COV_EXCL_STOP

    dev = device()
    let T = promote_type(T1, T2, T3)
        # only use the above kernel if we don't care about determinism
        # and if atomic operations are supported on these inputs
        atomic = if capability(device()) >= v"7.0"
            T <: Union{Int16, Int32, Int64, Float16, Float32, Float64}
        else
            T <: Union{Int32, Int64, Float32, Float64}
        end
        if CUDA.math_mode() == CUDA.PEDANTIC_MATH || !atomic
            bc = Base.Broadcast.broadcasted(A, Base.CartesianIndices(A)) do a, I
                i, j = Tuple(I)
                LinearAlgebra.dot(x[i], a * y[j])
            end
            return sum(bc)
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
        wanted_threads = shuffle ? nextwarp(dev, mA) : mA
        function compute_threads(max_threads)
            if wanted_threads > max_threads
                shuffle ? prevwarp(dev, max_threads) : max_threads
            else
                wanted_threads
            end
        end

        # how many threads can we launch?
        kernel_func = @cuda launch=false kernel(x, A, y, res, Val(shuffle))
        compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
        config = launch_configuration(kernel_func.fun; shmem=compute_shmem∘compute_threads)
        threads = compute_threads(config.threads)
        blocks = min(config.blocks, cld(mA, config.blocks))
        shmem = compute_shmem(threads)
        kernel_func(x, A, y, res, Val(shuffle); threads, blocks, shmem)

        CUDA.@allowscalar res[]
    end
end

function LinearAlgebra.:(*)(transx::Transpose{<:Any,<:StridedCuVector{T}},
                            y::StridedCuVector{T}) where T<:Union{ComplexF16, CublasComplex}
    x = transx.parent
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    return dotu(n, x, y)
end

function LinearAlgebra.norm(x::DenseCuArray{<:Union{Float16, ComplexF16, CublasFloat}},
                            p::Real=2)
    if p == 2
        return nrm2(x)
    else
        return invoke(norm, Tuple{AbstractGPUArray, Real}, x, p)
    end
end
LinearAlgebra.norm(x::Diagonal{T, <:StridedCuVector{T}}, p::Real=2) where {T<:Union{Float16, ComplexF16, CublasFloat}} = norm(x.diag, p)
LinearAlgebra.norm2(x::DenseCuArray{<:Union{Float16, ComplexF16, CublasFloat}}) = nrm2(x)

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
        return rmul!(Y, beta)
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

# work around upstream breakage from JuliaLang/julia#55547
@static if VERSION >= v"1.11.2"
    const CuUpperOrUnitUpperTriangular = LinearAlgebra.UpperOrUnitUpperTriangular{
        <:Any,<:Union{<:CuArray, Adjoint{<:Any, <:CuArray}, Transpose{<:Any, <:CuArray}}}
    const CuLowerOrUnitLowerTriangular = LinearAlgebra.LowerOrUnitLowerTriangular{
        <:Any,<:Union{<:CuArray, Adjoint{<:Any, <:CuArray}, Transpose{<:Any, <:CuArray}}}
    LinearAlgebra.istriu(::CuUpperOrUnitUpperTriangular) = true
    LinearAlgebra.istril(::CuUpperOrUnitUpperTriangular) = false
    LinearAlgebra.istriu(::CuLowerOrUnitLowerTriangular) = false
    LinearAlgebra.istril(::CuLowerOrUnitLowerTriangular) = true
end



#
# BLAS 3
#

# GEMM

@static if VERSION ≥ v"1.12.0-rc"
    function LinearAlgebra.generic_matmatmul_wrapper!(C::StridedCuMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}, alpha::Number, beta::Number, val::LinearAlgebra.BlasFlag.SyrkHerkGemm) where {T<:CublasFloat}
        LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, alpha, beta)
    end
    function LinearAlgebra._symm_hemm_generic!(C::StridedCuMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, alpha, beta, ::Val{BlasFlag.SYMM}) where {T}
        lrchar, ulchar = LinearAlgebra._lrchar_ulchar(tA, tB)
        if lrchar == 'L'
            symm!(lrchar, ulchar, alpha, A, B, beta, C)
        else
            symm!(lrchar, ulchar, alpha, B, A, beta, C)
        end
    end
    function LinearAlgebra._symm_hemm_generic!(C::StridedCuMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, alpha, beta, ::Val{BlasFlag.HEMM}) where {T}
        lrchar, ulchar = LinearAlgebra._lrchar_ulchar(tA, tB)
        if lrchar == 'L'
            hemm!(lrchar, ulchar, alpha, A, B, beta, C)
        else
            hemm!(lrchar, ulchar, alpha, B, A, beta, C)
        end
    end
end

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
        return LinearAlgebra.rmul!(C, beta)
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

# fallback for weird Complex edge cases
const AdjOrTransOrCuMatrix{T} = Union{StridedCuMatrix{T}, AdjOrTrans{<:T,<:StridedCuMatrix{T}}}
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::AdjOrTransOrCuMatrix{T}, B::Adjoint{T, <:Transpose{T, <:StridedCuMatrix{T}}}, α::Number, β::Number) where {T<:Complex} = mul!(C, A, conj(parent(parent(B))), α, β)
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::AdjOrTransOrCuMatrix{T}, B::Transpose{T, <:Adjoint{T, <:StridedCuMatrix{T}}}, α::Number, β::Number) where {T<:Complex} = mul!(C, A, conj(parent(parent(B))), α, β)
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::Adjoint{T, <:Transpose{T, <:StridedCuMatrix{T}}}, B::AdjOrTransOrCuMatrix{T}, α::Number, β::Number) where {T<:Complex} = mul!(C, conj(parent(parent(A))), B, α, β)
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::Transpose{T, <:Adjoint{T, <:StridedCuMatrix{T}}}, B::AdjOrTransOrCuMatrix{T}, α::Number, β::Number) where {T<:Complex} = mul!(C, conj(parent(parent(A))), B, α, β)

# triangular

LinearAlgebra.generic_trimatmul!(C::StridedCuMatrix{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trmm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
LinearAlgebra.generic_mattrimul!(C::StridedCuMatrix{T}, uploc, isunitc, tfun::Function, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trmm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, A, C)

## tri-tri-mul!
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

# conversions to dense matrices
Base.Array(D::Diagonal{T, <:CuArray{T}}) where {T} = Array(Diagonal(Array(D.diag)))
CUDA.CuArray(D::Diagonal{T}) where {T} = CuMatrix(D)
function CUDA.CuMatrix{T}(D::Diagonal) where {T}
    n = size(D, 1)
    B = CUDA.zeros(T, n, n)
    n == 0 && return B

    gpu_diag = adapt(CuArray, D.diag)
    ## COV_EXCL_START
    function fill_diagonal()
        i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        grid_stride = gridDim().x * blockDim().x
        while i <= n
            @inbounds B[i,i] = gpu_diag[i]
            i += grid_stride
        end
        return
    end
    ## COV_EXCL_STOP
    kernel = @cuda launch = false fill_diagonal()
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, n)
    blocks = min(config.blocks, cld(n, threads))
    @cuda threads blocks fill_diagonal()

    return B
end

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
    C .*= transpose(B.diag)
    return C
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Diagonal{T,<:CuVector}, B::Transpose{T,<:CuMatrix}) where {T<:CublasFloat}
    C .= B
    C .*= A.diag
    return C
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Adjoint{T,<:CuMatrix}, B::Diagonal{T,<:CuVector}) where {T<:CublasFloat}
    C .= A
    C .*= transpose(B.diag)
    return C
end

function LinearAlgebra.mul!(C::CuMatrix{T}, A::Diagonal{T,<:CuVector}, B::Adjoint{T,<:CuMatrix}) where {T<:CublasFloat}
    C .= B
    C .*= A.diag
    return C
end

function LinearAlgebra.mul!(C::Diagonal{T, <:CuVector}, A::Union{<:CuMatrix{T}, Adjoint{T, <:CuMatrix}, Transpose{T, <:CuMatrix}}, B::Union{<:CuMatrix{T}, Adjoint{T, <:CuMatrix}, Transpose{T, <:CuMatrix}}) where {T<:CublasFloat}
    Cfull   = A*B
    C.diag .= diag(Cfull)
    return C
end

function LinearAlgebra.lmul!(A::Diagonal{T,<:CuVector{T}}, B::CuMatrix{T}) where {T<:CublasFloat}
    return dgmm!('L', B, A.diag, B)
end

function LinearAlgebra.rmul!(A::CuMatrix{T}, B::Diagonal{T,<:CuVector{T}}) where {T<:CublasFloat}
    return dgmm!('R', A, B.diag, A)
end

# eltypes do not match
function LinearAlgebra.lmul!(A::Diagonal{T,<:CuVector{T}}, B::CuMatrix) where {T<:CublasFloat}
    @. B = A.diag * B
    return B
end
function LinearAlgebra.lmul!(A::Diagonal{Td,<:CuVector{Td}}, B::Transpose{Tt, <:CuMatrix{Tt}}) where {Td<:CublasFloat, Tt<:CublasFloat}
    @. B = A.diag * B
    return B
end
function LinearAlgebra.lmul!(A::Diagonal{Td,<:CuVector{Td}}, B::Adjoint{Tt, <:CuMatrix{Tt}}) where {Td<:CublasFloat, Tt<:CublasFloat}
    @. B = A.diag * B
    return B
end
# eltypes do not match
function LinearAlgebra.rmul!(A::CuMatrix, B::Diagonal{T,<:CuVector{T}}) where {T<:CublasFloat}
    At = transpose(A)
    @. At = B.diag * At
    return A
end
function LinearAlgebra.rmul!(A::Transpose{Tt, <:CuMatrix{Tt}}, B::Diagonal{Td,<:CuVector{Td}}) where {Td<:CublasFloat, Tt<:CublasFloat}
    At = parent(A)
    @. At = B.diag * At
    return transpose(At)
end
function LinearAlgebra.rmul!(A::Adjoint{Tt, <:CuMatrix{Tt}}, B::Diagonal{Td,<:CuVector{Td}}) where {Td<:CublasFloat, Tt<:CublasFloat}
    At = parent(A)
    @. At = adjoint(B.diag) * At
    return adjoint(At)
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
