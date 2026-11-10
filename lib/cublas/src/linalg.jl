# interfacing with LinearAlgebra standard library

using LinearAlgebra: MulAddMul, AdjOrTrans, wrap, UpperOrLowerTriangular, rmul!, lmul!

# BLAS walks a vector with a negative increment backwards from its end, while a reversed
# view points at its first element, so vector operands need a positive stride.
blas_layout(x::StridedCuVector) = stride(x, 1) > 0
blas_layout(A::StridedCuMatrix) = column_major(A)

# run a cuBLAS routine on contiguous copies of operands it can't address
blas_input(X) = blas_layout(X) ? X : copy(X)
blas_output(f, C) = blas_layout(C) ? f(C) : copyto!(C, f(similar(C)))

#
# BLAS 1
#

function LinearAlgebra.rmul!(x::CuArray{<:CublasFloat}, k::Bool)
    # explicitly fill x with zero to comply with julias "false = strong zero"
    !k && fill!(x, zero(eltype(x)))
    return x
end

# cuBLAS can only scale arrays that it can address as a vector
LinearAlgebra.rmul!(x::StridedCuArray{<:CublasFloat}, k::Number) =
  x isa StridedCuVecOrDenseMat ? scal!(length(x), k, x) : (x .*= k)

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

function LinearAlgebra.:(*)(transx::Transpose{<:Any,<:StridedCuVector{T}},
                            y::StridedCuVector{T}) where T<:Union{ComplexF16, CublasComplex}
    x = transx.parent
    n = length(x)
    n==length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    return dotu(n, x, y)
end

function LinearAlgebra.norm(x::StridedCuVecOrDenseMat{<:Union{Float16, ComplexF16, CublasFloat}},
                            p::Real=2)
    if p == 2
        return nrm2(x)
    else
        return invoke(norm, Tuple{AnyGPUArray, Real}, x, p)
    end
end
LinearAlgebra.norm(x::Diagonal{T, <:StridedCuVector{T}}, p::Real=2) where {T<:Union{Float16, ComplexF16, CublasFloat}} = norm(x.diag, p)
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

# `Givens{T}` stores the cosine as a `T`, so complex rotations carry a complex `c` with a
# zero imaginary part; cuBLAS takes a real one.
function LinearAlgebra.lmul!(G::LinearAlgebra.Givens, A::StridedCuVecOrMat{T}) where T<:CublasFloat
    if G.i2 > size(A, 1)
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    rotate!(view(A, G.i1, :), view(A, G.i2, :), convert(real(T), G.c), G.s)
    return A
end

function LinearAlgebra.rmul!(A::StridedCuMatrix{T}, G::LinearAlgebra.Givens) where T<:CublasFloat
    if G.i2 > size(A, 2)
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    rotate!(view(A, :, G.i1), view(A, :, G.i2), convert(real(T), G.c), -conj(G.s))
    return A
end



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

# triangular

## tri-tri-mul!
function LinearAlgebra.generic_trimatmul!(C::StridedCuMatrix{T}, uplocA, isunitcA, tfunA::Function, A::StridedCuMatrix{T}, triB::UpperOrLowerTriangular{T,<:AdjOrTransOrCuMatrix{T}}) where {T<:CublasFloat}
    uplocB = LinearAlgebra.uplo_char(triB)
    isunitcB = LinearAlgebra.isunit_char(triB)
    B = parent(triB)
    tfunB = LinearAlgebra.wrapperop(B)
    transa = tfunA === identity ? 'N' : tfunA === transpose ? 'T' : 'C'
    transb = tfunB === identity ? 'N' : tfunB === transpose ? 'T' : 'C'
    # one operand gets its other triangle zeroed so cuBLAS can treat it as a full matrix;
    # do so on a copy, which also gives cuBLAS a layout it can address
    blas_output(C) do C
        if uplocA == 'L' && tfunA === identity && tfunB === identity && uplocB == 'U' && isunitcB == 'N' # lower * upper
            trmm!('L', uplocA, transa, isunitcA, one(T), blas_input(A), triu!(copy(B)), C)
        elseif uplocA == 'U' && tfunA === identity && tfunB === identity && uplocB == 'L' && isunitcB == 'N' # upper * lower
            trmm!('L', uplocA, transa, isunitcA, one(T), blas_input(A), tril!(copy(B)), C)
        elseif uplocA == 'U' && tfunA === identity && tfunB !== identity && uplocB == 'U' && isunitcA == 'N'
            # operation is reversed to avoid executing the tranpose
            trmm!('R', uplocB, transb, isunitcB, one(T), blas_input(parent(B)), triu!(copy(A)), C)
        elseif uplocA == 'L' && tfunA !== identity && tfunB === identity && uplocB == 'L' && isunitcB == 'N'
            trmm!('L', uplocA, transa, isunitcA, one(T), blas_input(A), tril!(copy(B)), C)
        elseif uplocA == 'U' && tfunA !== identity && tfunB === identity && uplocB == 'U' && isunitcB == 'N'
            trmm!('L', uplocA, transa, isunitcA, one(T), blas_input(A), triu!(copy(B)), C)
        elseif uplocA == 'L' && tfunA === identity && tfunB !== identity && uplocB == 'L' && isunitcA == 'N'
            trmm!('R', uplocB, transb, isunitcB, one(T), blas_input(parent(B)), tril!(copy(A)), C)
        else
            throw("mixed triangular-triangular multiplication") # TODO: rethink
        end
    end
    return C
end

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

# Diagonal

# conversions to dense matrices
Base.Array(D::Diagonal{T, <:CuArray{T}}) where {T} = Array(Diagonal(Array(D.diag)))
CUDACore.CuArray(D::Diagonal{T}) where {T} = CuMatrix(D)
function CUDACore.CuMatrix{T}(D::Diagonal) where {T}
    n = size(D, 1)
    B = CUDACore.zeros(T, n, n)
    n == 0 && return B

    gpu_diag = adapt(CuArray, D.diag)
    function fill_diagonal()
        i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        grid_stride = gridDim().x * blockDim().x
        while i <= n
            @inbounds B[i,i] = gpu_diag[i]
            i += grid_stride
        end
        return
    end
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

# fast transpose using geam
function LinearAlgebra.transpose!(dest::CuMatrix{T}, src::CuMatrix{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    axes(dest) == reverse(axes(src)) || throw(DimensionMismatch("axes of the destination are incompatible with that of the source"))
    geam!('T', 'T', one(T), src, zero(T), src, dest)
    return dest
end
