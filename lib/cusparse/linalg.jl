using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, BlasInt

function LinearAlgebra.opnorm(A::CuSparseMatrixCSR, p::Real=2)
    if p == Inf
        return maximum(sum(abs, A; dims=2))
    elseif p == 1
        return maximum(sum(abs, A; dims=1))
    else
        throw(ArgumentError("p=$p is not supported"))
    end
end

LinearAlgebra.opnorm(A::CuSparseMatrixCSC, p::Real=2) = opnorm(CuSparseMatrixCSR(A), p)

function LinearAlgebra.norm(A::AbstractCuSparseMatrix{T}, p::Real=2) where T
    if p == Inf
        return maximum(abs.(A.nzVal))
    elseif p == -Inf
        return minimum(abs.(A.nzVal))
    elseif p == 0
        return Float64(A.nnz)
    else
        return sum(abs.(A.nzVal).^p)^(1/p)
    end
end

function LinearAlgebra.triu(A::CuSparseMatrixCOO, k::Integer=0)
    mask = A.rowInd .+ k .<= A.colInd
    rows = A.rowInd[mask]
    cols = A.colInd[mask]
    vals = A.nzVal[mask]
    sparse(rows, cols, vals, size(A)..., fmt = :coo)
end

function LinearAlgebra.tril(A::CuSparseMatrixCOO, k::Integer=0)
    mask = A.rowInd .+ k .>= A.colInd
    rows = A.rowInd[mask]
    cols = A.colInd[mask]
    vals = A.nzVal[mask]
    sparse(rows, cols, vals, size(A)..., fmt = :coo)
end

function SparseArrays.droptol!(A::CuSparseMatrixCOO, tol::Real)
    mask = abs.(A.nzVal) .> tol
    rows = A.rowInd[mask]
    cols = A.colInd[mask]
    vals = A.nzVal[mask]
    B = sparse(rows, cols, vals, size(A)..., fmt = :coo)
    copyto!(A, B)
end

function Base.reshape(A::CuSparseMatrixCOO, dims::Dims)
    nrows, ncols = size(A)
    flat_indices = nrows .* (A.colInd .- 1) .+ A.rowInd .- 1
    new_col, new_row = div.(flat_indices, dims[1]) .+ 1, rem.(flat_indices, dims[1]) .+ 1
    sparse(new_row, new_col, A.nzVal, dims[1], length(dims) == 1 ? 1 : dims[2], fmt = :coo)
end

function LinearAlgebra.kron(A::CuSparseMatrixCOO{T, Ti}, B::CuSparseMatrixCOO{T, Ti}) where {Ti, T}
    mA,nA = size(A)
    mB,nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Annz = Int64(A.nnz)
    Bnnz = Int64(B.nnz)

    if Annz == 0 || Bnnz == 0
        return CuSparseMatrixCOO(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{T}(undef, 0), out_shape)
    end

    row = (A.rowInd .- 1) .* mB
    row = repeat(row, inner = Bnnz)
    col = (A.colInd .- 1) .* nB
    col = repeat(col, inner = Bnnz)
    data = repeat(A.nzVal, inner = Bnnz)

    row .+= repeat(B.rowInd .- 1, outer = Annz) .+ 1
    col .+= repeat(B.colInd .- 1, outer = Annz) .+ 1

    data .*= repeat(B.nzVal, outer = Annz)

    sparse(row, col, data, out_shape..., fmt = :coo)
end

function LinearAlgebra.kron(A::CuSparseMatrixCOO{T, Ti}, B::Diagonal{TB}) where {Ti, T, TB}
    mA,nA = size(A)
    mB,nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Annz = Int64(A.nnz)
    Bnnz = nB

    if Annz == 0 || Bnnz == 0
        return CuSparseMatrixCOO(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{T}(undef, 0), out_shape)
    end

    row = (A.rowInd .- 1) .* mB
    row = repeat(row, inner = Bnnz)
    col = (A.colInd .- 1) .* nB
    col = repeat(col, inner = Bnnz)
    data = repeat(A.nzVal, inner = Bnnz)

    row .+= CuVector(repeat(0:nB-1, outer = Annz)) .+ 1
    col .+= CuVector(repeat(0:nB-1, outer = Annz)) .+ 1

    Bdiag = (TB == Bool) ? CUDA.ones(T, nB) : B.diag
    data .*= repeat(Bdiag, outer = Annz)

    sparse(row, col, data, out_shape..., fmt = :coo)
end

function LinearAlgebra.kron(A::Diagonal{TA}, B::CuSparseMatrixCOO{T, Ti}) where {Ti, T, TA}
    mA,nA = size(A)
    mB,nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Annz = nA
    Bnnz = Int64(B.nnz)

    if Annz == 0 || Bnnz == 0
        return CuSparseMatrixCOO(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{T}(undef, 0), out_shape)
    end

    row = (0:nA-1) .* mB
    row = CuVector(repeat(row, inner = Bnnz))
    col = (0:nA-1) .* nB
    col = CuVector(repeat(col, inner = Bnnz))
    Adiag = (TA == Bool) ? CUDA.ones(T, nA) : A.diag
    data = repeat(Adiag, inner = Bnnz)

    row .+= repeat(B.rowInd .- 1, outer = Annz) .+ 1
    col .+= repeat(B.colInd .- 1, outer = Annz) .+ 1

    data .*= repeat(B.nzVal, outer = Annz)

    sparse(row, col, data, out_shape..., fmt = :coo)
end

function LinearAlgebra.dot(y::CuVector{T}, A::CuSparseMatrixCSC{T}, x::CuVector{T}) where {T<:Union{BlasInt, BlasFloat}}
    if length(y) != size(A, 1) || length(x) != size(A, 2)
        throw(DimensionMismatch("dimensions must match"))
    end
    n = size(A, 2)

    ## COV_EXCL_START
    function kernel(y::CuDeviceVector{T1}, colPtr::CuDeviceVector{T2}, rowVal::CuDeviceVector{T2},
        nzVal::CuDeviceVector{T1}, x::CuDeviceVector{T1}, result::CuDeviceVector{T1}, n::Integer, shuffle) where {T1,T2}

        thread_idx = threadIdx().x
        index = (blockIdx().x-1) * blockDim().x + thread_idx
        stride = blockDim().x * gridDim().x

        tmp = zero(T1)
        if index <=n
            @inbounds for col in index:stride:n
                for j in (colPtr[col]):(colPtr[col+1]-1)
                    row = rowVal[j]
                    val = nzVal[j]
                    tmp += dot(y[row], val, x[col])
                end
            end
        end

        reduced_val = CUDA.reduce_block(+, tmp, zero(T1), shuffle)

        if thread_idx == 1
            @inbounds result[blockIdx().x] = reduced_val
        end
        return
    end
    ## COV_EXCL_STOP

    function compute_threads(max_threads, wanted_threads, shuffle, dev)
        if wanted_threads > max_threads
            shuffle ? prevwarp(dev, max_threads) : max_threads
        else
            wanted_threads
        end
    end

    shuffle = true

    result = CuArray{T}(undef, 1)
    kernel = @cuda launch=false kernel(y, A.colPtr, A.rowVal, A.nzVal, x, result, n, Val(shuffle))
    config = launch_configuration(kernel.fun)
    threads = compute_threads(config.threads, n, shuffle, device())
    blocks = min(config.blocks, cld(n, threads))
    result = CuArray{T}(undef, blocks)
    kernel(y, A.colPtr, A.rowVal, A.nzVal, x, result, n, Val(shuffle); threads, blocks)

    return sum(result)
end

function LinearAlgebra.dot(y::CuVector{T}, A::CuSparseMatrixCSR{T}, x::CuVector{T}) where {T<:Union{BlasInt, BlasFloat}}
    if length(y) != size(A, 1) || length(x) != size(A, 2)
        throw(DimensionMismatch("dimensions must match"))
    end
    n = size(A, 1)

    ## COV_EXCL_START
    function kernel(y::CuDeviceVector{T1}, rowPtr::CuDeviceVector{T2}, colVal::CuDeviceVector{T2},
        nzVal::CuDeviceVector{T1}, x::CuDeviceVector{T1}, result::CuDeviceVector{T1}, n::Integer, shuffle) where {T1,T2}

        thread_idx = threadIdx().x
        index = (blockIdx().x-1) * blockDim().x + thread_idx
        stride = blockDim().x * gridDim().x

        tmp = zero(T1)
        if index <= n
            @inbounds for row in index:stride:n
                for j in rowPtr[row]:(rowPtr[row+1]-1)
                    col = colVal[j]
                    val = nzVal[j]
                    tmp += dot(y[row], val, x[col])
                end
            end
        end

        reduced_val = CUDA.reduce_block(+, tmp, zero(T1), shuffle)

        if thread_idx == 1
            @inbounds result[blockIdx().x] = reduced_val
        end
        return
    end
    ## COV_EXCL_STOP

    function compute_threads(max_threads, wanted_threads, shuffle, dev)
        if wanted_threads > max_threads
            shuffle ? prevwarp(dev, max_threads) : max_threads
        else
            wanted_threads
        end
    end

    shuffle = true

    result = CuArray{T}(undef, 1)
    kernel = @cuda launch=false kernel(y, A.rowPtr, A.colVal, A.nzVal, x, result, n, Val(shuffle))
    config = launch_configuration(kernel.fun)
    threads = compute_threads(config.threads, n, shuffle, device())
    blocks = min(config.blocks, cld(n, threads))
    result = CuArray{T}(undef, blocks)
    kernel(y, A.rowPtr, A.colVal, A.nzVal, x, result, n, Val(shuffle); threads, blocks)

    return sum(result)
end

# work around upstream breakage from JuliaLang/julia#55547
@static if VERSION >= v"1.11.2"
    const CuSparseUpperOrUnitUpperTriangular = LinearAlgebra.UpperOrUnitUpperTriangular{
        <:Any,<:Union{<:AbstractCuSparseMatrix, Adjoint{<:Any, <:AbstractCuSparseMatrix}, Transpose{<:Any, <:AbstractCuSparseMatrix}}}
    const CuSparseLowerOrUnitLowerTriangular = LinearAlgebra.LowerOrUnitLowerTriangular{
        <:Any,<:Union{<:AbstractCuSparseMatrix, Adjoint{<:Any, <:AbstractCuSparseMatrix}, Transpose{<:Any, <:AbstractCuSparseMatrix}}}
    LinearAlgebra.istriu(::CuSparseUpperOrUnitUpperTriangular) = true
    LinearAlgebra.istril(::CuSparseUpperOrUnitUpperTriangular) = false
    LinearAlgebra.istriu(::CuSparseLowerOrUnitLowerTriangular) = false
    LinearAlgebra.istril(::CuSparseLowerOrUnitLowerTriangular) = true
end

for SparseMatrixType in [:CuSparseMatrixCSC, :CuSparseMatrixCSR]
    @eval begin
        LinearAlgebra.triu(A::$SparseMatrixType{T}, k::Integer) where {T} =
            $SparseMatrixType( triu(CuSparseMatrixCOO(A), k) )
        LinearAlgebra.triu(A::Transpose{T,<:$SparseMatrixType}, k::Integer) where {T} =
            $SparseMatrixType( triu(CuSparseMatrixCOO(_sptranspose(parent(A))), k) )
        LinearAlgebra.triu(A::Adjoint{T,<:$SparseMatrixType}, k::Integer) where {T} =
            $SparseMatrixType( triu(CuSparseMatrixCOO(_spadjoint(parent(A))), k) )

        LinearAlgebra.tril(A::$SparseMatrixType{T}, k::Integer) where {T} =
            $SparseMatrixType( tril(CuSparseMatrixCOO(A), k) )
        LinearAlgebra.tril(A::Transpose{T,<:$SparseMatrixType}, k::Integer) where {T} =
            $SparseMatrixType( tril(CuSparseMatrixCOO(_sptranspose(parent(A))), k) )
        LinearAlgebra.tril(A::Adjoint{T,<:$SparseMatrixType}, k::Integer) where {T} =
            $SparseMatrixType( tril(CuSparseMatrixCOO(_spadjoint(parent(A))), k) )

        LinearAlgebra.triu(A::Union{$SparseMatrixType{T}, Transpose{T,<:$SparseMatrixType}, Adjoint{T,<:$SparseMatrixType}}) where {T} =
            $SparseMatrixType( triu(CuSparseMatrixCOO(A), 0) )
        LinearAlgebra.tril(A::Union{$SparseMatrixType{T},Transpose{T,<:$SparseMatrixType}, Adjoint{T,<:$SparseMatrixType}}) where {T} =
            $SparseMatrixType( tril(CuSparseMatrixCOO(A), 0) )

        LinearAlgebra.kron(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(A), CuSparseMatrixCOO(B)) )
        LinearAlgebra.kron(A::$SparseMatrixType{T}, B::Diagonal) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(A), B) )
        LinearAlgebra.kron(A::Diagonal, B::$SparseMatrixType{T}) where {T} =
            $SparseMatrixType( kron(A, CuSparseMatrixCOO(B)) )

        LinearAlgebra.kron(A::Transpose{T,<:$SparseMatrixType}, B::$SparseMatrixType{T}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(_sptranspose(parent(A))), CuSparseMatrixCOO(B)) )
        LinearAlgebra.kron(A::$SparseMatrixType{T}, B::Transpose{T,<:$SparseMatrixType}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(A), CuSparseMatrixCOO(_sptranspose(parent(B)))) )
        LinearAlgebra.kron(A::Transpose{T,<:$SparseMatrixType}, B::Transpose{T,<:$SparseMatrixType}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(_sptranspose(parent(A))), CuSparseMatrixCOO(_sptranspose(parent(B)))) )
        LinearAlgebra.kron(A::Transpose{T,<:$SparseMatrixType}, B::Diagonal) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(_sptranspose(parent(A))), B) )
        LinearAlgebra.kron(A::Diagonal, B::Transpose{T,<:$SparseMatrixType}) where {T} =
            $SparseMatrixType( kron(A, CuSparseMatrixCOO(_sptranspose(parent(B)))) )

        LinearAlgebra.kron(A::Adjoint{T,<:$SparseMatrixType}, B::$SparseMatrixType{T}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(_spadjoint(parent(A))), CuSparseMatrixCOO(B)) )
        LinearAlgebra.kron(A::$SparseMatrixType{T}, B::Adjoint{T,<:$SparseMatrixType}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(A), CuSparseMatrixCOO(_spadjoint(parent(B)))) )
        LinearAlgebra.kron(A::Adjoint{T,<:$SparseMatrixType}, B::Adjoint{T,<:$SparseMatrixType}) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(_spadjoint(parent(A))), CuSparseMatrixCOO(_spadjoint(parent(B)))) )
        LinearAlgebra.kron(A::Adjoint{T,<:$SparseMatrixType}, B::Diagonal) where {T} =
            $SparseMatrixType( kron(CuSparseMatrixCOO(_spadjoint(parent(A))), B) )
        LinearAlgebra.kron(A::Diagonal, B::Adjoint{T,<:$SparseMatrixType}) where {T} =
            $SparseMatrixType( kron(A, CuSparseMatrixCOO(_spadjoint(parent(B)))) )


        function Base.reshape(A::$SparseMatrixType, dims::Dims)
            B = CuSparseMatrixCOO(A)
            $SparseMatrixType(reshape(B, dims))
        end

        function SparseArrays.droptol!(A::$SparseMatrixType, tol::Real)
            B = CuSparseMatrixCOO(A)
            droptol!(B, tol)
            copyto!(A, $SparseMatrixType(B))
        end

        function LinearAlgebra.exp(A::$SparseMatrixType; threshold = 1e-7, nonzero_tol = 1e-14)
            rows = LinearAlgebra.checksquare(A) # Throws exception if not square
            typeA = eltype(A)

            mat_norm = norm(A, Inf)
            scaling_factor = nextpow(2, mat_norm) # Native routine, faster
            A = A ./ scaling_factor
            delta = 1

            P = $SparseMatrixType(spdiagm(0 => ones(eltype(A), rows)))
            next_term = P
            n = 1

            while delta > threshold
                next_term = typeA(1 / n) * A * next_term
                droptol!(next_term, nonzero_tol)
                delta = norm(next_term, Inf)
                copyto!(P, P + next_term)
                n = n + 1
            end
            for n = 1:log2(scaling_factor)
                P = P * P;
                if nnz(P) / length(P) < 0.25
                    droptol!(P, nonzero_tol)
                end
            end
            P
        end
    end
end
