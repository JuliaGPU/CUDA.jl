# interfacing with LinearAlgebra standard library

using LinearAlgebra: MulAddMul, AdjOrTrans, wrap, UpperOrLowerTriangular, rmul!, lmul!

#
# BLAS 1
#

# dot for element types cuBLAS doesn't support
function GPUArrays.blas_dot(x::AnyCuArray{T1}, y::AnyCuArray{T2}) where {T1,T2}
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))

    n == 0 && return zero(promote_type(T1, T2))
    # custom kernel using simple linear indexing and atomic additions,
    # resulting in about 10% speed-up compared to a simple mapreduce.
    function kernel(x, y, res::AbstractArray{T}, shuffle) where {T}
        local_val = zero(T)

        # grid-stride loop. The index is an `Int`, as an `Int32` one would wrap around
        # (and keep looping, or read out of bounds) for arrays near `typemax(Int32)`.
        i = Int(threadIdx().x + (blockIdx().x - 1i32)*blockDim().x)
        while i <= length(x)
            @inbounds local_val += LinearAlgebra.dot(x[i], y[i])
            i += blockDim().x * gridDim().x
        end

        val = CUDACore.reduce_block(+, local_val, zero(T), shuffle)
        if threadIdx().x == 1i32
            # NOTE: introduces nondeterminism
            @inbounds CUDACore.@atomic res[] += val
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
        if CUDACore.math_mode() == CUDACore.PEDANTIC_MATH || !atomic
            return mapreduce((x,y)->LinearAlgebra.dot(x, y), +, x, y; init=zero(T))
        end

        res = CUDACore.zeros(T, 1)

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
        compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
        call = CUDACore.KernelCall(kernel, x, y, res, Val(shuffle))
        compiled = CUDACore.kernel_compile(call)
        config = launch_configuration(compiled.fun; shmem=compute_shmem∘compute_threads)
        threads = compute_threads(config.threads)
        blocks = min(config.blocks, cld(n, threads))
        shmem = compute_shmem(threads)
        CUDACore.kernel_launch(compiled, call; threads, blocks, shmem)

        CUDACore.@allowscalar res[]
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
    function kernel(x, A, y, res::AbstractArray{T}, shuffle) where {T}
        local_val = zero(T)

        # grid-stride loop over rows, with an `Int` index as above
        i = Int(threadIdx().x + (blockIdx().x - 1i32)*blockDim().x)
        while i <= mA
            row_val = zero(T)
            for j in 1:nA
                # XXX: this is slow, but the focus is on avoiding materializing A*y
                @inbounds row_val += A[i, j] * y[j]
            end
            @inbounds local_val += LinearAlgebra.dot(x[i], row_val)
            i += blockDim().x * gridDim().x
        end

        val = CUDACore.reduce_block(+, local_val, zero(T), shuffle)
        if threadIdx().x == 1i32
            # NOTE: introduces nondeterminism
            @inbounds CUDACore.@atomic res[] += val
        end

        return
    end

    dev = device()
    let T = promote_type(T1, T2, T3)
        # only use the above kernel if we don't care about determinism
        # and if atomic operations are supported on these inputs
        atomic = if capability(device()) >= v"7.0"
            T <: Union{Int16, Int32, Int64, Float16, Float32, Float64}
        else
            T <: Union{Int32, Int64, Float32, Float64}
        end
        if CUDACore.math_mode() == CUDACore.PEDANTIC_MATH || !atomic
            bc = Base.Broadcast.broadcasted(A, Base.CartesianIndices(A)) do a, I
                i, j = Tuple(I)
                LinearAlgebra.dot(x[i], a * y[j])
            end
            # Unlike sum, dot must not widen small-integer results.
            return mapreduce(identity, +, bc)
        end

        # a grid without blocks can't be launched
        mA == 0 && return zero(T)

        res = CUDACore.zeros(T, 1)

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
        compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
        call = CUDACore.KernelCall(kernel, x, A, y, res, Val(shuffle))
        compiled = CUDACore.kernel_compile(call)
        config = launch_configuration(compiled.fun; shmem=compute_shmem∘compute_threads)
        threads = compute_threads(config.threads)
        blocks = min(config.blocks, cld(mA, threads))
        shmem = compute_shmem(threads)
        CUDACore.kernel_launch(compiled, call; threads, blocks, shmem)

        CUDACore.@allowscalar res[]
    end
end

LinearAlgebra.norm2(x::StridedCuVecOrDenseMat{<:Union{Float16, ComplexF16, CublasFloat}}) = nrm2(x)

# keep the 2-norm on the GPU, so that normalizing doesn't wait for it to be copied back.
# Unlike LinearAlgebra on Julia 1.14, this doesn't rescale and recompute a subnormal norm,
# which would cost two more passes over `x` for every call.
function LinearAlgebra.normalize!(x::StridedCuVecOrDenseMat{T}, p::Real=2) where {T<:Union{Float16, CublasFloat}}
    if p == 2
        nrm = CuArray{real(T)}(undef)
        nrm2(x, nrm)
        x ./= nrm
        return x
    else
        return invoke(normalize!, Tuple{AbstractArray, Real}, x, p)
    end
end
LinearAlgebra.normalize(x::StridedCuVecOrDenseMat{T}, p::Real=2) where {T<:Union{Float16, CublasFloat}} =
    normalize!(copy(x), p)

#
# BLAS 2
#

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

# fallback for weird Complex edge cases
const AdjOrTransOrCuMatrix{T} = Union{StridedCuMatrix{T}, AdjOrTrans{<:T,<:StridedCuMatrix{T}}}
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::AdjOrTransOrCuMatrix{T}, B::Adjoint{T, <:Transpose{T, <:StridedCuMatrix{T}}}, α::Number, β::Number) where {T<:Complex} = mul!(C, A, conj(parent(parent(B))), α, β)
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::AdjOrTransOrCuMatrix{T}, B::Transpose{T, <:Adjoint{T, <:StridedCuMatrix{T}}}, α::Number, β::Number) where {T<:Complex} = mul!(C, A, conj(parent(parent(B))), α, β)
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::Adjoint{T, <:Transpose{T, <:StridedCuMatrix{T}}}, B::AdjOrTransOrCuMatrix{T}, α::Number, β::Number) where {T<:Complex} = mul!(C, conj(parent(parent(A))), B, α, β)
LinearAlgebra.mul!(C::StridedCuVecOrMat{T}, A::Transpose{T, <:Adjoint{T, <:StridedCuMatrix{T}}}, B::AdjOrTransOrCuMatrix{T}, α::Number, β::Number) where {T<:Complex} = mul!(C, conj(parent(parent(A))), B, α, β)

# GPUArrays implements LinearAlgebra's matrix products and triangular operations, and calls
# these hooks for the operations that BLAS supports.

GPUArrays.blas_gemv!(tA::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T}, x::StridedCuVector{T},
                     β::Union{Bool,T}, y::StridedCuVector{T}) where {T<:CublasFloat} =
    gemv!(tA, α, A, x, β, y)
GPUArrays.blas_symv!(uplo::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T}, x::StridedCuVector{T},
                     β::Union{Bool,T}, y::StridedCuVector{T}) where {T<:CublasFloat} =
    symv!(uplo, α, A, x, β, y)
GPUArrays.blas_hemv!(uplo::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T}, x::StridedCuVector{T},
                     β::Union{Bool,T}, y::StridedCuVector{T}) where {T<:CublasComplex} =
    hemv!(uplo, α, A, x, β, y)
GPUArrays.blas_trmv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     A::StridedCuMatrix{T}, x::StridedCuVector{T}) where {T<:CublasFloat} =
    trmv!(uplo, trans, diag, A, x)
GPUArrays.blas_trsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     A::StridedCuMatrix{T}, x::StridedCuVector{T}) where {T<:CublasFloat} =
    trsv!(uplo, trans, diag, A, x)

# gemm! also implements mixed-precision products with gemmEx
function GPUArrays.blas_gemm!(tA::AbstractChar, tB::AbstractChar, α::Number, A::StridedCuVecOrMat,
                              B::StridedCuVecOrMat, β::Number, C::StridedCuVecOrMat)
    T = eltype(C)
    m, k = size(A, tA == 'N' ? 1 : 2), size(A, tA == 'N' ? 2 : 1)
    n = size(B, tB == 'N' ? 2 : 1)
    if gemmExComputeType(eltype(A), eltype(B), T, m, k, n) !== nothing ||
       T <: CublasFloat && eltype(A) == eltype(B) == T
        return gemm!(tA, tB, α, A, B, β, C)
    end
    return GPUArrays.generic_matmatmul!(C, tA, tB, A, B, α, β)
end
GPUArrays.blas_symm!(side::AbstractChar, uplo::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T},
                     B::StridedCuMatrix{T}, β::Union{Bool,T}, C::StridedCuMatrix{T}) where {T<:CublasFloat} =
    symm!(side, uplo, α, A, B, β, C)
GPUArrays.blas_hemm!(side::AbstractChar, uplo::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T},
                     B::StridedCuMatrix{T}, β::Union{Bool,T}, C::StridedCuMatrix{T}) where {T<:CublasComplex} =
    hemm!(side, uplo, α, A, B, β, C)
GPUArrays.blas_trmm!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     α::Union{Bool,T}, A::StridedCuMatrix{T}, B::StridedCuMatrix{T},
                     C::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trmm!(side, uplo, trans, diag, α, A, B, C)
GPUArrays.blas_trsm!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     α::Union{Bool,T}, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T<:CublasFloat} =
    trsm!(side, uplo, trans, diag, α, A, B)
GPUArrays.blas_syrk!(uplo::AbstractChar, trans::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T},
                     β::Union{Bool,T}, C::StridedCuMatrix{T}) where {T<:CublasFloat} =
    syrk!(uplo, trans, α, A, β, C)
GPUArrays.blas_herk!(uplo::AbstractChar, trans::AbstractChar, α::Real, A::StridedCuMatrix{T},
                     β::Real, C::StridedCuMatrix{T}) where {T<:CublasComplex} =
    herk!(uplo, trans, α, A, β, C)
GPUArrays.blas_geam!(tA::AbstractChar, tB::AbstractChar, α::Union{Bool,T}, A::StridedCuMatrix{T},
                     β::Union{Bool,T}, B::StridedCuMatrix{T}, C::StridedCuMatrix{T}) where {T<:CublasFloat} =
    geam!(tA, tB, α, A, β, B, C)
GPUArrays.blas_dgmm!(side::AbstractChar, A::StridedCuMatrix{T}, x::StridedCuVector{T},
                     C::StridedCuMatrix{T}) where {T<:CublasFloat} =
    dgmm!(side, A, x, C)

GPUArrays.blas_scal!(α::Number, x::StridedCuVecOrDenseMat{<:CublasFloat}) = scal!(length(x), α, x)
LinearAlgebra.dot(A::Symmetric{<:Union{Real,Complex},<:CuMatrix}, B::Symmetric{<:Union{Real,Complex},<:CuMatrix}) =
    GPUArrays.gpu_dot(A, B)
LinearAlgebra.dot(A::Hermitian{<:Union{Real,Complex},<:CuMatrix}, B::Hermitian{<:Union{Real,Complex},<:CuMatrix}) =
    GPUArrays.gpu_dot(A, B)
GPUArrays.blas_dot(x::StridedCuVecOrDenseMat{T}, y::StridedCuVecOrDenseMat{T}) where {T<:Union{Float16, CublasReal}} =
    dot(length(x), x, y)
GPUArrays.blas_dot(x::StridedCuVecOrDenseMat{T}, y::StridedCuVecOrDenseMat{T}) where {T<:Union{ComplexF16, CublasComplex}} =
    dotc(length(x), x, y)
GPUArrays.blas_dotu(x::StridedCuVector{T}, y::StridedCuVector{T}) where {T<:Union{ComplexF16, CublasComplex}} =
    dotu(length(x), x, y)
GPUArrays.blas_nrm2(x::StridedCuVecOrDenseMat{<:Union{Float16, ComplexF16, CublasFloat}}) = nrm2(x)
GPUArrays.blas_asum(x::StridedCuVecOrDenseMat{<:CublasFloat}) = asum(length(x), x)
GPUArrays.blas_axpy!(α::Number, x::StridedCuVecOrDenseMat{T}, y::StridedCuVecOrDenseMat{T}) where {T<:Union{Float16, ComplexF16, CublasFloat}} =
    axpy!(length(x), α, x, y)
GPUArrays.blas_axpby!(α::Number, x::StridedCuVecOrDenseMat{T}, β::Number, y::StridedCuVecOrDenseMat{T}) where {T<:Union{Float16, ComplexF16, CublasFloat}} =
    axpby!(length(x), α, x, β, y)
GPUArrays.blas_rot!(x::StridedCuVecOrDenseMat{T}, y::StridedCuVecOrDenseMat{T}, c::Real, s::Number) where {T<:CublasFloat} =
    rot!(length(x), x, y, c, s)
