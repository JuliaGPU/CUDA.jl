using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, BlasInt

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

_kron_CuSparseMatrixCOO_components(A::CuSparseMatrixCOO) = A.rowInd, A.colInd, A.nzVal, identity, Int(A.nnz)
_kron_CuSparseMatrixCOO_components(At::Transpose{<:Number, <:CuSparseMatrixCOO}) = parent(At).colInd, parent(At).rowInd, parent(At).nzVal, transpose, Int(parent(At).nnz)
_kron_CuSparseMatrixCOO_components(Ah::Adjoint{<:Number, <:CuSparseMatrixCOO}) = parent(Ah).colInd, parent(Ah).rowInd, parent(Ah).nzVal, adjoint, Int(parent(Ah).nnz)

function LinearAlgebra.kron(
    A::Union{CuSparseMatrixCOO{TvA, TiA}, Transpose{TvA, <:CuSparseMatrixCOO{TvA, TiA}}, Adjoint{TvA, <:CuSparseMatrixCOO{TvA, TiA}}},
    B::Union{CuSparseMatrixCOO{TvB, TiB}, Transpose{TvB, <:CuSparseMatrixCOO{TvB, TiB}}, Adjoint{TvB, <:CuSparseMatrixCOO{TvB, TiB}}}
    ) where {TvA, TiA, TvB, TiB}
    mA, nA = size(A)
    mB, nB = size(B)
    Ti = promote_type(TiA, TiB)
    Tv = typeof(oneunit(TvA)*oneunit(TvB))

    A_rowInd, A_colInd, A_nzVal, A_nzOp, A_nnz = _kron_CuSparseMatrixCOO_components(A)
    B_rowInd, B_colInd, B_nzVal, B_nzOp, B_nnz = _kron_CuSparseMatrixCOO_components(B)

    if A_nnz == 0 || B_nnz == 0
        return CuSparseMatrixCOO{Tv, Ti}(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{Tv}(undef, 0), (mA * mB, nA * nB))
    end

    C_nnz = A_nnz * B_nnz
    C_rowInd = reshape(B_rowInd .+ Ti(mB) .* (reshape(A_rowInd, (1, A_nnz)) .- one(Ti)), C_nnz)
    C_colInd = reshape(B_colInd .+ Ti(nB) .* (reshape(A_colInd, (1, A_nnz)) .- one(Ti)), C_nnz)
    C_nzVal = reshape(B_nzOp.(B_nzVal) .* A_nzOp.(reshape(A_nzVal, (1, A_nnz))), C_nnz)

    C = CuSparseMatrixCOO{Tv, Ti}(C_rowInd, C_colInd, C_nzVal, (mA * mB, nA * nB), C_nnz)
    return sort_coo(C)
end

function LinearAlgebra.kron(
    A::Union{CuSparseMatrixCOO{TvA, TiA}, Transpose{TvA, <:CuSparseMatrixCOO{TvA, TiA}}, Adjoint{TvA, <:CuSparseMatrixCOO{TvA, TiA}}},
    B::Diagonal{TvB, <:Union{CuVector{TvB}, Base.ReshapedArray{TvB, 1, <:Adjoint{TvB, <:CuVector{TvB}}}}}
    ) where {TvA, TiA, TvB}
    mA, nA = size(A)
    mB, nB = size(B)
    Ti = TiA
    Tv = typeof(oneunit(TvA)*oneunit(TvB))

    A_rowInd, A_colInd, A_nzVal, A_nzOp, A_nnz = _kron_CuSparseMatrixCOO_components(A)
    B_rowInd, B_colInd, B_nzVal, B_nnz = one(Ti):Ti(nB), one(Ti):Ti(nB), B.diag, Int(nB)

    if A_nnz == 0 || B_nnz == 0
        return CuSparseMatrixCOO{Tv, Ti}(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{Tv}(undef, 0), (mA * mB, nA * nB))
    end

    C_nnz = A_nnz * B_nnz
    C_rowInd = reshape(B_rowInd .+ Ti(mB) .* (reshape(A_rowInd, (1, A_nnz)) .- one(Ti)), C_nnz)
    C_colInd = reshape(B_colInd .+ Ti(nB) .* (reshape(A_colInd, (1, A_nnz)) .- one(Ti)), C_nnz)
    C_nzVal = reshape(B_nzVal .* A_nzOp.(reshape(A_nzVal, (1, A_nnz))), C_nnz)

    C = CuSparseMatrixCOO{Tv, Ti}(C_rowInd, C_colInd, C_nzVal, (mA * mB, nA * nB), C_nnz)
    return sort_coo(C)
end

function LinearAlgebra.kron(
    A::Diagonal{TvA, <:Union{CuVector{TvA}, Base.ReshapedArray{TvA, 1, <:Adjoint{TvA, <:CuVector{TvA}}}}},
    B::Union{CuSparseMatrixCOO{TvB, TiB}, Transpose{TvB, <:CuSparseMatrixCOO{TvB, TiB}}, Adjoint{TvB, <:CuSparseMatrixCOO{TvB, TiB}}}
    ) where {TvA, TvB, TiB}
    mA, nA = size(A)
    mB, nB = size(B)
    Ti = TiB
    Tv = typeof(oneunit(TvA)*oneunit(TvB))

    A_rowInd, A_colInd, A_nzVal, A_nnz = one(Ti):Ti(nA), one(Ti):Ti(nA), A.diag, Int(nA)
    B_rowInd, B_colInd, B_nzVal, B_nzOp, B_nnz = _kron_CuSparseMatrixCOO_components(B)

    if A_nnz == 0 || B_nnz == 0
        return CuSparseMatrixCOO{Tv, Ti}(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{Tv}(undef, 0), (mA * mB, nA * nB))
    end

    C_nnz = A_nnz * B_nnz
    C_rowInd = reshape(B_rowInd .+ Ti(mB) .* (reshape(A_rowInd, (1, A_nnz)) .- one(Ti)), C_nnz)
    C_colInd = reshape(B_colInd .+ Ti(nB) .* (reshape(A_colInd, (1, A_nnz)) .- one(Ti)), C_nnz)
    C_nzVal = reshape(B_nzOp.(B_nzVal) .* reshape(A_nzVal, (1, A_nnz)), C_nnz)

    C = CuSparseMatrixCOO{Tv, Ti}(C_rowInd, C_colInd, C_nzVal, (mA * mB, nA * nB), C_nnz)
    return sort_coo(C)
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

