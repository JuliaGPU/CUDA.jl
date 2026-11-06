# conversions involving cuSPARSE-specific functionality
#
# Conversions between the canonical formats, and between dense and sparse arrays, are
# GPUArrays' portable implementations. What remains here are the cuSPARSE routines for
# the BSR format, pruning, and sorting the buffers of arrays that were constructed from
# unsorted buffers.

export sort_csc, sort_csr, sort_coo

"""
    sparse(x::CuMatrix; fmt=:csc)

Return a sparse CUDA matrix with `Cint` indices, with type determined by `fmt`: `:csc`,
`:csr`, `:coo` or `:bsr`.
"""
function SparseArrays.sparse(A::CuMatrix; fmt::Symbol=:csc)
    # (with `Cint` indices, like the constructors)
    fmt === :csc && return CuSparseMatrixCSC(A)
    fmt === :csr && return CuSparseMatrixCSR(A)
    fmt === :coo && return CuSparseMatrixCOO(A)
    fmt === :bsr && return CuSparseMatrixBSR(A)
    throw(ArgumentError("Format :$fmt not available, use :csc, :csr, :bsr or :coo."))
end

# The sorting and format conversion routines of cuSPARSE only support 32-bit indices.
# Matrices with other index types are narrowed to `Cint` around these calls, and the
# results are widened back to the index type of the input.
function check_cint_indices(op, sizes::Integer...)
    for sz in sizes
        sz <= typemax(Cint) || throw(ArgumentError(
            "$op uses 32-bit indices in cuSPARSE, which cannot represent a size of $sz"))
    end
    return
end

cint_indices(x::CuVector) = convert(CuVector{Cint}, x)

function sort_csc(A::CuSparseMatrixCSC{Tv,Ti}, index::SparseChar='O') where {Tv,Ti}

    m,n = size(A)
    check_cint_indices("sort_csc", m, n, nnz(A) + 1)
    perm = CuArray{Cint}(undef, nnz(A))
    cusparseCreateIdentityPermutation(handle(), nnz(A), perm)

    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    sorted_colPtr = CuVector{Cint}(A.colPtr)
    sorted_rowVal = CuVector{Cint}(A.rowVal)
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseXcscsort_bufferSizeExt(handle(), m, n, nnz(A), sorted_colPtr, sorted_rowVal, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseXcscsort(handle(), m, n, nnz(A), descA, sorted_colPtr, sorted_rowVal, perm, buffer)
    end
    perm .+= one(Cint)
    sorted_nzVal = A.nzVal[perm]
    CUDACore.unsafe_free!(perm)
    CuSparseMatrixCSC{Tv,Ti}(sorted_colPtr, sorted_rowVal, sorted_nzVal, size(A))
end

function sort_csr(A::CuSparseMatrixCSR{Tv,Ti}, index::SparseChar='O') where {Tv,Ti}

    m,n = size(A)
    (iszero(m) || iszero(n)) && return A
    check_cint_indices("sort_csr", m, n, nnz(A) + 1)
    perm = CuArray{Cint}(undef, nnz(A))
    cusparseCreateIdentityPermutation(handle(), nnz(A), perm)

    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    sorted_rowPtr = CuVector{Cint}(A.rowPtr)
    sorted_colVal = CuVector{Cint}(A.colVal)
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseXcsrsort_bufferSizeExt(handle(), m, n, nnz(A), sorted_rowPtr, sorted_colVal, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseXcsrsort(handle(), m, n, nnz(A), descA, sorted_rowPtr, sorted_colVal, perm, buffer)
    end
    perm .+= one(Cint)
    sorted_nzVal = A.nzVal[perm]
    CUDACore.unsafe_free!(perm)
    CuSparseMatrixCSR{Tv,Ti}(sorted_rowPtr, sorted_colVal, sorted_nzVal, size(A))
end

# (column-sorted COO matrices are not valid `CuSparseMatrixCOO`s, which are sorted by row)
function sort_coo(A::CuSparseMatrixCOO{Tv,Ti}, type::SparseChar='R') where {Tv,Ti}
    type == 'R' || throw(ArgumentError("type=$type was used and only type='R' is supported."))

    m,n = size(A)
    check_cint_indices("sort_coo", m, n, nnz(A))
    perm = CuArray{Cint}(undef, nnz(A))
    cusparseCreateIdentityPermutation(handle(), nnz(A), perm)

    sorted_rowInd = CuVector{Cint}(A.rowInd)
    sorted_colInd = CuVector{Cint}(A.colInd)
    function bufferSize()
        # It seems that in some cases `out` is not updated
        # and we have the following error in the tests:
        # "Out of GPU memory trying to allocate 127.781 TiB".
        # We set 0 as default value to avoid it.
        out = Ref{Csize_t}(1)
        cusparseXcoosort_bufferSizeExt(handle(), m, n, nnz(A), sorted_rowInd, sorted_colInd, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseXcoosortByRow(handle(), m, n, nnz(A), sorted_rowInd, sorted_colInd, perm, buffer)
    end

    perm .+= one(Cint)
    sorted_nzVal = A.nzVal[perm]
    CUDACore.unsafe_free!(perm)
    CuSparseMatrixCOO{Tv,Ti}(sorted_rowInd, sorted_colInd, sorted_nzVal, size(A))
end

for (bname, fname, pname, elty) in ((:cusparseSpruneCsr2csr_bufferSizeExt, :cusparseSpruneCsr2csrNnz, :cusparseSpruneCsr2csr, :Float32),
                                    (:cusparseDpruneCsr2csr_bufferSizeExt, :cusparseDpruneCsr2csrNnz, :cusparseDpruneCsr2csr, :Float64))
    @eval begin
        function prune(A::CuSparseMatrixCSR{$elty, Ti}, threshold::Number, index::SparseChar) where {Ti}
            m, n = size(A)
            check_cint_indices("prune", m, n, nnz(A) + 1)
            rowPtr = cint_indices(A.rowPtr)
            colVal = cint_indices(A.colVal)
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            descC = CuMatrixDescriptor('G', 'L', 'N', index)
            rowPtrC = CuVector{Int32}(undef, m+1)

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), m, n, nnz(A), descA, nonzeros(A), rowPtr, colVal,
                       Ref{$elty}(threshold), descC, CuPtr{$elty}(CU_NULL), rowPtrC, CuPtr{Int32}(CU_NULL), out)
                return out[]
            end

            return with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                $fname(handle(), m, n, nnz(A), descA, nonzeros(A), rowPtr, colVal,
                       Ref{$elty}(threshold), descC, rowPtrC, nnzTotal, buffer)

                colValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $pname(handle(), m, n, nnz(A), descA, nonzeros(A), rowPtr, colVal,
                       Ref{$elty}(threshold), descC, nzValC, rowPtrC, colValC, buffer)
                CuSparseMatrixCSR{$elty, Ti}(rowPtrC, colValC, nzValC, (m, n))
            end
        end

        function prune(A::CuSparseMatrixCSC{$elty, Ti}, threshold::Number, index::SparseChar) where {Ti}
            m, n = size(A)
            check_cint_indices("prune", m, n, nnz(A) + 1)
            colPtr = cint_indices(A.colPtr)
            rowVal = cint_indices(A.rowVal)
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            descC = CuMatrixDescriptor('G', 'L', 'N', index)
            colPtrC = CuVector{Int32}(undef, n+1)

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), n, m, nnz(A), descA, nonzeros(A), colPtr, rowVal,
                       Ref{$elty}(threshold), descC, CuPtr{$elty}(CU_NULL), colPtrC, CuPtr{Int32}(CU_NULL), out)
                return out[]
            end

            return with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                $fname(handle(), n, m, nnz(A), descA, nonzeros(A), colPtr, rowVal,
                       Ref{$elty}(threshold), descC, colPtrC, nnzTotal, buffer)

                rowValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $pname(handle(), n, m, nnz(A), descA, nonzeros(A), colPtr, rowVal,
                       Ref{$elty}(threshold), descC, nzValC, colPtrC, rowValC, buffer)
                CuSparseMatrixCSC{$elty, Ti}(colPtrC, rowValC, nzValC, (m, n))
            end
        end
    end
end

## CSR to BSR and vice-versa

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :ComplexF32),
                     (:cusparseZcsr2bsr, :ComplexF64))
    @eval begin
        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty, Ti}, blockDim::Integer;
                                          dir::SparseChar='R', index::SparseChar='O',
                                          indc::SparseChar='O') where {Ti}
            m,n = size(csr)
            mb = cld(m, blockDim)
            nb = cld(n, blockDim)
            # cusparseXcsr2bsrNnz rejects zero-dim inputs with CUSPARSE_STATUS_INVALID_VALUE;
            # the result is unambiguously empty, so construct it directly.
            if iszero(m) || iszero(n)
                bsrRowPtr = CuArray{Ti}(undef, mb + 1)
                bsrRowPtr .= (indc == 'O' ? one(Ti) : zero(Ti))
                bsrColInd = CuArray{Ti}(undef, 0)
                bsrNzVal  = CuArray{$elty}(undef, 0)
                return CuSparseMatrixBSR{$elty, Ti}(bsrRowPtr, bsrColInd, bsrNzVal, size(csr), blockDim, dir, 0)
            end
            check_cint_indices("CSR to BSR conversion", m, n, nnz(csr) + 1)
            csrRowPtr = cint_indices(csr.rowPtr)
            csrColInd = cint_indices(csr.colVal)
            nnz_ref = Ref{Cint}(1)
            bsrRowPtr = CUDACore.zeros(Cint, mb + 1)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', index)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            cusparseXcsr2bsrNnz(handle(), dir, m, n, cudesca, csrRowPtr,
                                csrColInd, blockDim, cudescc, bsrRowPtr, nnz_ref)
            (nnz_ref[] > mb * nb) && error("The number of nonzero blocks of the BSR matrix is incorrect.")
            bsrNzVal = CUDACore.zeros($elty, nnz_ref[] * blockDim * blockDim )
            bsrColInd = CUDACore.zeros(Cint, nnz_ref[])
            $fname(handle(), dir, m, n,
                   cudesca, nonzeros(csr), csrRowPtr, csrColInd,
                   blockDim, cudescc, bsrNzVal, bsrRowPtr,
                   bsrColInd)
            # (with the index type of the input; the BSR routines use a `Cint` view)
            CuSparseMatrixBSR{$elty, Ti}(bsrRowPtr, bsrColInd, bsrNzVal, size(csr), blockDim, dir, nnz_ref[])
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :ComplexF32),
                     (:cusparseZbsr2csr, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty, Ti};
                                          index::SparseChar='O', indc::SparseChar='O') where {Ti}
            m,n = size(bsr)
            mb = cld(m, bsr.blockDim)
            nb = cld(n, bsr.blockDim)
            check_cint_indices("BSR to CSR conversion", widemul(mb, bsr.blockDim),
                               widemul(nb, bsr.blockDim), widemul(bsr.nnzb, widemul(bsr.blockDim, bsr.blockDim)) + 1)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', index)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            # the result covers whole blocks, which can extend past the matrix
            padded_m, padded_n = mb*Int(bsr.blockDim), nb*Int(bsr.blockDim)
            csrRowPtr = CuArray{Cint}(undef, padded_m + 1)
            # cusparse*bsr2csr does not initialize csrRowPtr when the block grid
            # is empty; pre-fill with the empty-matrix sentinel.
            if iszero(mb) || iszero(nb)
                csrRowPtr .= (indc == 'O' ? one(Cint) : zero(Cint))
            end
            csrColInd = CUDACore.zeros(Cint, nnz(bsr))
            csrNzVal  = CUDACore.zeros($elty, nnz(bsr))
            $fname(handle(), bsr.dir, mb, nb,
                   cudesca, nonzeros(bsr), cint_indices(bsr.rowPtr), cint_indices(bsr.colVal),
                   bsr.blockDim, cudescc, csrNzVal, csrRowPtr,
                   csrColInd)
            csr = CuSparseMatrixCSR{$elty, Ti}(csrRowPtr, csrColInd, csrNzVal, (padded_m, padded_n))
            (padded_m, padded_n) == (m, n) ? csr : csr[1:m, 1:n]
        end
    end
end

# implement Int conversions using reinterpreted Float
for (elty, felty) in ((:Int16, :Float16),
                      (:Int32, :Float32),
                      (:Int64, :Float64),
                      (:Int128, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty, Ti};
                                          index::SparseChar='O', indc::SparseChar='O') where {Ti}
            bsr_compat = CuSparseMatrixBSR{$felty, Ti}(
                bsr.rowPtr,
                bsr.colVal,
                reinterpret($felty, bsr.nzVal),
                size(bsr),
                bsr.blockDim,
                bsr.dir,
                bsr.nnzb
            )
            csr_compat = CuSparseMatrixCSR{$felty}(bsr_compat; index, indc)
            CuSparseMatrixCSR{$elty, Ti}(
                csr_compat.rowPtr,
                csr_compat.colVal,
                reinterpret($elty, csr_compat.nzVal),
                size(csr_compat)
            )
        end

        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty, Ti}, blockDim) where {Ti}
            csr_compat = CuSparseMatrixCSR{$felty, Ti}(
                csr.rowPtr,
                csr.colVal,
                reinterpret($felty, csr.nzVal),
                size(csr)
            )
            bsr_compat = CuSparseMatrixBSR(csr_compat, blockDim)
            CuSparseMatrixBSR{$elty, Ti}(
                bsr_compat.rowPtr,
                bsr_compat.colVal,
                reinterpret($elty, bsr_compat.nzVal),
                size(bsr_compat),
                bsr_compat.blockDim,
                bsr_compat.dir,
                bsr_compat.nnzb
            )
        end
    end
end
CuSparseMatrixBSR(coo::CuSparseMatrixCOO, blockdim) = CuSparseMatrixBSR(CuSparseMatrixCSR(coo), blockdim) # no direct conversion
CuSparseMatrixBSR(csc::CuSparseMatrixCSC, blockdim) = CuSparseMatrixBSR(CuSparseMatrixCSR(csc), blockdim) # no direct conversion
CuSparseMatrixBSR(csr::CuSparseMatrixCSR{T}, blockdim; kwargs...) where {T} =
    CuSparseMatrixBSR{T}(csr, blockdim; kwargs...)
CuSparseMatrixBSR{T,Ti}(A::CuCanonicalMatrix, blockdim::Integer; kwargs...) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(CuSparseMatrixBSR(CuSparseMatrixCSR{T}(A), blockdim; kwargs...))

CuSparseMatrixCSR(bsr::CuSparseMatrixBSR{T}; kwargs...) where {T} = CuSparseMatrixCSR{T}(bsr; kwargs...)
CuSparseMatrixCSR{T,Ti}(bsr::CuSparseMatrixBSR) where {T,Ti} =
    GPUSparseMatrixCSR{T,Ti}(CuSparseMatrixCSR(bsr))
for S in (:CuSparseMatrixCSC, :CuSparseMatrixCOO)
    @eval begin
        $S(bsr::CuSparseMatrixBSR) = $S(CuSparseMatrixCSR(bsr)) # no direct conversion
        $S{T}(bsr::CuSparseMatrixBSR) where {T} = $S{T}(CuSparseMatrixCSR(bsr))
        $S{T,Ti}(bsr::CuSparseMatrixBSR) where {T,Ti} = $S{T,Ti}(CuSparseMatrixCSR(bsr))
    end
end

function CUDACore.CuMatrix{T}(bsr::CuSparseMatrixBSR{T, Ti}; index::SparseChar='O',
                          indc::SparseChar='O') where {T, Ti}
    CuMatrix{T}(CuSparseMatrixCSR{T}(bsr; index, indc))
end

function CuSparseMatrixBSR(A::CuMatrix, blockDim::Integer=max(gcd(size(A)...), 1); index::SparseChar='O')
    CuSparseMatrixBSR(CuSparseMatrixCSR{eltype(A),Cint}(A), blockDim; index)
end
