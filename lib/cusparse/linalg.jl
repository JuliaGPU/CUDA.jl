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

function LinearAlgebra.kron(A::CuSparseMatrixCOO{T1, Ti}, B::Diagonal{T2, <:CuVector{T2}}) where {Ti, T1, T2}
    T = promote_type(T1, T2)
    mA,nA = size(A)
    mB,nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Annz = Int64(A.nnz)

    # Find non-zero diagonal elements
    nz_mask = map(!iszero, B.diag)
    nz_indices = findall(nz_mask)
    Bnnz = length(nz_indices)

    if Annz == 0 || Bnnz == 0
        return CuSparseMatrixCOO(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{T}(undef, 0), out_shape)
    end

    # Only process non-zero diagonal elements
    nz_diag = B.diag[nz_indices]
    nz_offsets = CuVector(nz_indices .- 1)

    row = (A.rowInd .- 1) .* mB
    row = repeat(row, inner = Bnnz)
    col = (A.colInd .- 1) .* nB
    col = repeat(col, inner = Bnnz)
    data = repeat(convert(CuVector{T}, A.nzVal), inner = Bnnz)

    row .+= repeat(nz_offsets, outer = Annz) .+ 1
    col .+= repeat(nz_offsets, outer = Annz) .+ 1

    data .*= repeat(nz_diag, outer = Annz)

    sparse(row, col, data, out_shape..., fmt = :coo)
end

function LinearAlgebra.kron(A::Diagonal{T1, <:CuVector{T1}}, B::CuSparseMatrixCOO{T2, Ti}) where {Ti, T1, T2}
    T = promote_type(T1, T2)
    mA,nA = size(A)
    mB,nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Bnnz = Int64(B.nnz)

    # Find non-zero diagonal elements
    nz_mask = map(!iszero, A.diag)
    nz_indices = findall(nz_mask)
    Annz = length(nz_indices)

    if Bnnz == 0 || Annz == 0
        return CuSparseMatrixCOO(CuVector{Ti}(undef, 0), CuVector{Ti}(undef, 0), CuVector{T}(undef, 0), out_shape)
    end

    # Only process non-zero diagonal elements
    nz_diag = A.diag[nz_indices]
    nz_offsets = CuVector(nz_indices .- 1)

    row = nz_offsets .* mB
    row = repeat(row, inner = Bnnz)
    col = nz_offsets .* nB
    col = repeat(col, inner = Bnnz)
    data = repeat(convert(CuVector{T}, nz_diag), inner = Bnnz)

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

