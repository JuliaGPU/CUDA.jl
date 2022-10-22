# conversion routines between different sparse and dense storage formats

"""
    sparse(x::DenseCuMatrix; fmt=:csc)
    sparse(I::CuVector, J::CuVector, V::CuVector, [m, n]; fmt=:csc)

Return a sparse cuda matrix, with type determined by `fmt`.
Possible formats are :csc, :csr, :bsr, and :coo.
"""
function SparseArrays.sparse(x::DenseCuMatrix; fmt=:csc)
    if fmt == :csc
        return CuSparseMatrixCSC(x)
    elseif fmt == :csr
        return CuSparseMatrixCSR(x)
    elseif fmt == :bsr
        return CuSparseMatrixBSR(x)
    elseif fmt == :coo
        return CuSparseMatrixCOO(x)
    else
        error("Format :$fmt not available, use :csc, :csr, :bsr or :coo.")
    end
end

SparseArrays.sparse(I::CuVector, J::CuVector, V::CuVector; kws...) =
    sparse(I, J, V, maximum(I), maximum(J); kws...)

SparseArrays.sparse(I::CuVector, J::CuVector, V::CuVector, m, n; kws...) =
    sparse(Cint.(I), Cint.(J), V, m, n; kws...)

function SparseArrays.sparse(I::CuVector{Cint}, J::CuVector{Cint}, V::CuVector{Tv}, m, n;
                             fmt=:csc, sorted::Bool=false) where Tv
    coo = CuSparseMatrixCOO{Tv}(I, J, V, (m, n))

    # The COO format is assumed to be sorted by row.
    if !sorted
        coo = sort_rows(coo)
    end

    if fmt == :csc
        return CuSparseMatrixCSC(coo)
    elseif fmt == :csr
        return CuSparseMatrixCSR(coo)
    elseif fmt == :coo
        return coo
    else
        error("Format :$fmt not available, use :csc, :csr, or :coo.")
    end
end

function sort_rows(coo::CuSparseMatrixCOO{Tv,Ti}) where {Tv <: BlasFloat,Ti}
    m,n = size(coo)

    perm = CuArray{Ti}(undef, nnz(coo))
    cusparseCreateIdentityPermutation(handle(), nnz(coo), perm)

    sorted_rowInd = copy(coo.rowInd)
    sorted_colInd = copy(coo.colInd)
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseXcoosort_bufferSizeExt(handle(), m, n, nnz(coo), coo.rowInd,
            coo.colInd, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseXcoosortByRow(handle(), m, n, nnz(coo), sorted_rowInd, sorted_colInd, perm, buffer)
    end

    sorted_nzVal = similar(coo.nzVal)
    let spvec = CuSparseVector(perm, sorted_nzVal, nnz(coo))
        if version() >= v"11.3"
            gather!(spvec, nonzeros(coo), 'Z')
        else
            gthr!(spvec, nonzeros(coo), 'Z')
        end
    end

    CUDA.unsafe_free!(perm)
    CuSparseMatrixCOO{Tv}(sorted_rowInd, sorted_colInd, sorted_nzVal, size(coo))
end
function sort_rows(coo::CuSparseMatrixCOO{Tv,Ti}) where {Tv,Ti}
    perm = sortperm(coo.rowInd)
    sorted_rowInd = coo.rowInd[perm]
    sorted_colInd = coo.colInd[perm]
    sorted_nzVal = coo.nzVal[perm]
    CUDA.unsafe_free!(perm)

    CuSparseMatrixCOO{Tv}(sorted_rowInd, sorted_colInd, sorted_nzVal, size(coo))
end


## CSR to CSC

function CuSparseMatrixCSC{T}(S::Transpose{T, <:CuSparseMatrixCSR{T}}) where T
    csr = parent(S)
    return CuSparseMatrixCSC{T}(csr.rowPtr, csr.colVal, csr.nzVal, size(csr))
end

function CuSparseMatrixCSR{T}(S::Transpose{T, <:CuSparseMatrixCSC{T}}) where T
    csc = parent(S)
    return CuSparseMatrixCSR{T}(csc.colPtr, csc.rowVal, csc.nzVal, size(csc))
end

function CuSparseMatrixCSC{T}(S::Adjoint{T, <:CuSparseMatrixCSR{T}}) where {T <: Real}
    csr = parent(S)
    return CuSparseMatrixCSC{T}(csr.rowPtr, csr.colVal, csr.nzVal, size(csr))
end

function CuSparseMatrixCSR{T}(S::Adjoint{T, <:CuSparseMatrixCSC{T}}) where {T <: Real}
    csc = parent(S)
    return CuSparseMatrixCSR{T}(csc.colPtr, csc.rowVal, csc.nzVal, size(csc))
end

function CuSparseMatrixCSC{T}(S::Adjoint{T, <:CuSparseMatrixCSR{T}}) where {T <: Complex}
    csr = parent(S)
    return CuSparseMatrixCSC{T}(csr.rowPtr, csr.colVal, conj.(csr.nzVal), size(csr))
end

function CuSparseMatrixCSR{T}(S::Adjoint{T, <:CuSparseMatrixCSC{T}}) where {T <: Complex}
    csc = parent(S)
    return CuSparseMatrixCSR{T}(csc.colPtr, csc.rowVal, conj.(csc.nzVal), size(csc))
end

for SparseMatrixType in [:CuSparseMatrixCSC, :CuSparseMatrixCSR]
    @eval begin
        $SparseMatrixType(S::Diagonal) = $SparseMatrixType(cu(S))
        $SparseMatrixType(S::Diagonal{T, <:CuArray}) where T = $SparseMatrixType{T}(S)
        $SparseMatrixType{Tv}(S::Diagonal{T, <:CuArray}) where {Tv, T} = $SparseMatrixType{Tv, Cint}(S)
        function $SparseMatrixType{Tv, Ti}(S::Diagonal{T, <:CuArray}) where {Tv, Ti, T}
            m = size(S, 1)
            return $SparseMatrixType{Tv, Ti}(CuVector(1:(m+1)), CuVector(1:m), Tv.(S.diag), (m, m))
        end
    end
end


# by flipping rows and columns, we can use that to get CSC to CSR too
for elty in (Float32, Float64, ComplexF32, ComplexF64)
    @eval begin
        CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty, Cint}; inda::SparseChar='O') =
            CuSparseMatrixCSC{$elty, Cint}(csr; inda)
        CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty, Cint}; inda::SparseChar='O') =
            CuSparseMatrixCSR{$elty, Cint}(csc; inda)

        function CuSparseMatrixCSC{$elty, Cint}(csr::CuSparseMatrixCSR{$elty, Cint}; inda::SparseChar='O')
            m,n = size(csr)
            colPtr = CUDA.zeros(Cint, n+1)
            rowVal = CUDA.zeros(Cint, nnz(csr))
            nzVal = CUDA.zeros($elty, nnz(csr))
            # TODO: algorithm configuratibility?
            function bufferSize()
                out = Ref{Csize_t}(1)
                cusparseCsr2cscEx2_bufferSize(handle(), m, n, nnz(csr), nonzeros(csr),
                    csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                    $elty, CUSPARSE_ACTION_NUMERIC, inda,
                    CUSPARSE_CSR2CSC_ALG1, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                cusparseCsr2cscEx2(handle(), m, n, nnz(csr), nonzeros(csr),
                    csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                    $elty, CUSPARSE_ACTION_NUMERIC, inda,
                    CUSPARSE_CSR2CSC_ALG1, buffer)
            end
            CuSparseMatrixCSC{$elty, Cint}(colPtr,rowVal,nzVal,size(csr))
        end

        function CuSparseMatrixCSR{$elty, Cint}(csc::CuSparseMatrixCSC{$elty, Cint}; inda::SparseChar='O')
            m,n    = size(csc)
            rowPtr = CUDA.zeros(Cint,m+1)
            colVal = CUDA.zeros(Cint,nnz(csc))
            nzVal  = CUDA.zeros($elty,nnz(csc))
            # TODO: algorithm configuratibility?
            function bufferSize()
                out = Ref{Csize_t}(1)
                cusparseCsr2cscEx2_bufferSize(handle(), n, m, nnz(csc), nonzeros(csc),
                    csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                    $elty, CUSPARSE_ACTION_NUMERIC, inda,
                    CUSPARSE_CSR2CSC_ALG1, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                cusparseCsr2cscEx2(handle(), n, m, nnz(csc), nonzeros(csc),
                    csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                    $elty, CUSPARSE_ACTION_NUMERIC, inda,
                    CUSPARSE_CSR2CSC_ALG1, buffer)
            end
            CuSparseMatrixCSR{$elty, Cint}(rowPtr,colVal,nzVal,size(csc))
        end
    end
end

# implement Float16 conversions using wider types
# TODO: Float16 is sometimes natively supported
for (elty, welty) in ((:Float16, :Float32),
                      (:ComplexF16, :ComplexF32))
    @eval begin
        CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty, Cint}; inda::SparseChar='O') =
            CuSparseMatrixCSC{$elty, Cint}(csr; inda)
        CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty, Cint}; inda::SparseChar='O') =
            CuSparseMatrixCSR{$elty, Cint}(csc; inda)

        function CuSparseMatrixCSC{$elty, Cint}(csr::CuSparseMatrixCSR{$elty, Cint}; inda::SparseChar='O')
            m,n = size(csr)
            colPtr = CUDA.zeros(Cint, n+1)
            rowVal = CUDA.zeros(Cint, nnz(csr))
            nzVal = CUDA.zeros($elty, nnz(csr))
            # TODO: algorithm configuratibility?
            if $elty == Float16 #broken for ComplexF16?
                function bufferSize()
                    out = Ref{Csize_t}(1)
                    cusparseCsr2cscEx2_bufferSize(handle(), m, n, nnz(csr), nonzeros(csr),
                        csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                        $elty, CUSPARSE_ACTION_NUMERIC, inda,
                        CUSPARSE_CSR2CSC_ALG1, out)
                    return out[]
                end
                with_workspace(bufferSize) do buffer
                    cusparseCsr2cscEx2(handle(), m, n, nnz(csr), nonzeros(csr),
                        csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                        $elty, CUSPARSE_ACTION_NUMERIC, inda,
                        CUSPARSE_CSR2CSC_ALG1, buffer)
                end
                return CuSparseMatrixCSC{$elty, Cint}(colPtr,rowVal,nzVal,size(csr))
            else
                wide_csr = CuSparseMatrixCSR{$welty, Cint}(csr.rowPtr, csr.colVal, convert(CuVector{$welty}, nonzeros(csr)), size(csr))
                wide_csc = CuSparseMatrixCSC{$welty, Cint}(wide_csr)
                return CuSparseMatrixCSC{$elty, Cint}(wide_csc.colPtr, wide_csc.rowVal, convert(CuVector{$elty}, nonzeros(wide_csc)), size(wide_csc))
            end
        end

        function CuSparseMatrixCSR{$elty, Cint}(csc::CuSparseMatrixCSC{$elty, Cint}; inda::SparseChar='O')
            m,n    = size(csc)
            rowPtr = CUDA.zeros(Cint,m+1)
            colVal = CUDA.zeros(Cint,nnz(csc))
            nzVal  = CUDA.zeros($elty,nnz(csc))
            if $elty == Float16 #broken for ComplexF16?
                # TODO: algorithm configuratibility?
                function bufferSize()
                    out = Ref{Csize_t}(1)
                    cusparseCsr2cscEx2_bufferSize(handle(), n, m, nnz(csc), nonzeros(csc),
                        csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                        $elty, CUSPARSE_ACTION_NUMERIC, inda,
                        CUSPARSE_CSR2CSC_ALG1, out)
                    return out[]
                end
                with_workspace(bufferSize) do buffer
                    cusparseCsr2cscEx2(handle(), n, m, nnz(csc), nonzeros(csc),
                        csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                        $elty, CUSPARSE_ACTION_NUMERIC, inda,
                        CUSPARSE_CSR2CSC_ALG1, buffer)
                end
                return CuSparseMatrixCSR{$elty, Cint}(rowPtr,colVal,nzVal,size(csc))
            else
                wide_csc = CuSparseMatrixCSC{$welty, Cint}(csc.colPtr, csc.rowVal, convert(CuVector{$welty}, nonzeros(csc)), size(csc))
                wide_csr = CuSparseMatrixCSR{$welty, Cint}(wide_csc)
                return CuSparseMatrixCSR{$elty, Cint}(wide_csr.rowPtr, wide_csr.colVal, convert(CuVector{$elty}, nonzeros(wide_csr)), size(wide_csr))
            end
        end
    end
end

# implement Int conversions using reinterpreted Float
for (elty, felty) in ((:Int16, :Float16),
                      (:Int32, :Float32),
                      (:Int64, :Float64),
                      (:Int128, :ComplexF64))
    @eval begin
        CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty, Cint}) =
            CuSparseMatrixCSR{$elty, Cint}(csc)
        CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty, Cint}) =
            CuSparseMatrixCSC{$elty, Cint}(csr)
        function CuSparseMatrixCSR{$elty, Cint}(csc::CuSparseMatrixCSC{$elty, Cint})
            csc_compat = CuSparseMatrixCSC{$felty, Cint}(
                csc.colPtr,
                csc.rowVal,
                reinterpret($felty, csc.nzVal),
                size(csc)
            )
            csr_compat = CuSparseMatrixCSR(csc_compat)
            CuSparseMatrixCSR{$elty, Cint}(
                csr_compat.rowPtr,
                csr_compat.colVal,
                reinterpret($elty, csr_compat.nzVal),
                size(csr_compat)
            )
        end

        function CuSparseMatrixCSC{$elty, Cint}(csr::CuSparseMatrixCSR{$elty, Cint})
            csr_compat = CuSparseMatrixCSR{$felty, Cint}(
                csr.rowPtr,
                csr.colVal,
                reinterpret($felty, csr.nzVal),
                size(csr)
            )
            csc_compat = CuSparseMatrixCSC(csr_compat)
            CuSparseMatrixCSC{$elty, Cint}(
                csc_compat.colPtr,
                csc_compat.rowVal,
                reinterpret($elty, csc_compat.nzVal),
                size(csc_compat)
            )
        end
    end
end

## CSR to BSR and vice-versa

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :ComplexF32),
                     (:cusparseZcsr2bsr, :ComplexF64))
    @eval begin
        CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty, Cint}, blockDim::Integer;
            dir::SparseChar='R', inda::SparseChar='O', indc::SparseChar='O') =
                CuSparseMatrixBSR{$elty, Cint}(csr, blockDim; dir, inda, indc)

        function CuSparseMatrixBSR{$elty, Cint}(csr::CuSparseMatrixCSR{$elty, Cint}, blockDim::Integer;
                                          dir::SparseChar='R', inda::SparseChar='O',
                                          indc::SparseChar='O')
            m,n = size(csr)
            nnz_ref = Ref{Cint}(1)
            mb = cld(m, blockDim)
            nb = cld(n, blockDim)
            bsrRowPtr = CUDA.zeros(Cint,mb + 1)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', inda)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            cusparseXcsr2bsrNnz(handle(), dir, m, n, cudesca, csr.rowPtr,
                                csr.colVal, blockDim, cudescc, bsrRowPtr, nnz_ref)
            (nnz_ref[] > mb * nb) && error("The number of nonzero blocks of the BSR matrix is incorrect.")
            bsrNzVal = CUDA.zeros($elty, nnz_ref[] * blockDim * blockDim )
            bsrColInd = CUDA.zeros(Cint, nnz_ref[])
            $fname(handle(), dir, m, n,
                   cudesca, nonzeros(csr), csr.rowPtr, csr.colVal,
                   blockDim, cudescc, bsrNzVal, bsrRowPtr,
                   bsrColInd)
            CuSparseMatrixBSR{$elty, Cint}(bsrRowPtr, bsrColInd, bsrNzVal, size(csr), blockDim, dir, nnz_ref[])
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :ComplexF32),
                     (:cusparseZbsr2csr, :ComplexF64))
    @eval begin
        CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty, Cint};
                                          inda::SparseChar='O', indc::SparseChar='O') =
            CuSparseMatrixCSR{$elty, Cint}(bsr;inda, indc)
        function CuSparseMatrixCSR{$elty, Cint}(bsr::CuSparseMatrixBSR{$elty, Cint};
                                          inda::SparseChar='O', indc::SparseChar='O')
            m,n = size(bsr)
            mb = cld(m, bsr.blockDim)
            nb = cld(n, bsr.blockDim)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', inda)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            csrRowPtr = CUDA.zeros(Cint, m + 1)
            csrColInd = CUDA.zeros(Cint, nnz(bsr))
            csrNzVal  = CUDA.zeros($elty, nnz(bsr))
            $fname(handle(), bsr.dir, mb, nb,
                   cudesca, nonzeros(bsr), bsr.rowPtr, bsr.colVal,
                   bsr.blockDim, cudescc, csrNzVal, csrRowPtr,
                   csrColInd)
            # XXX: the size here may not match the expected size, when the matrix dimension
            #      is not a multiple of the block dimension!
            CuSparseMatrixCSR{$elty, Cint}(csrRowPtr, csrColInd, csrNzVal, (mb*bsr.blockDim, nb*bsr.blockDim))
        end
    end
end

# implement Int conversions using reinterpreted Float
for (elty, felty) in ((:Int16, :Float16),
                      (:Int32, :Float32),
                      (:Int64, :Float64),
                      (:Int128, :ComplexF64))
    @eval begin
        CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty, Cint}) = CuSparseMatrixCSR{$elty, Cint}(bsr)
        CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty, Cint}, blockDim) = CuSparseMatrixBSR{$elty, Cint}(csr, blockDim)

        function CuSparseMatrixCSR{$elty, Cint}(bsr::CuSparseMatrixBSR{$elty, Cint})
            bsr_compat = CuSparseMatrixBSR{$elty, Cint}(
                bsr.rowPtr,
                bsr.colVal,
                reinterpret($felty, bsr.nzVal),
                bsr.blockDim,
                bsr.dir,
                bsr.nnzb,
                size(bsr)
            )
            csr_compat = CuSparseMatrixCSR{$elty, Cint}(bsr_compat)
            CuSparseMatrixCSR{$elty, Cint}(
                csr_compat.rowPtr,
                csr_compat.colVal,
                reinterpret($elty, csr_compat.nzVal),
                size(csr_compat)
            )
        end

        function CuSparseMatrixBSR{$elty, Cint}(csr::CuSparseMatrixCSR{$elty, Cint}, blockDim)
            csr_compat = CuSparseMatrixCSR{$elty, Cint}(
                csr.rowPtr,
                csr.colVal,
                reinterpret($felty, csr.nzVal),
                size(csr)
            )
            bsr_compat = CuSparseMatrixBSR{$elty, Cint}(csr_compat, blockDim)
            CuSparseMatrixBSR{$elty, Cint}(
                bsr_compat.rowPtr,
                bsr_compat.colVal,
                reinterpret($elty, bsr_compat.nzVal),
                bsr_compat.blockDim,
                bsr_compat.dir,
                bsr_compat.nnzb,
                size(bsr_compat)
            )
        end
    end
end

## CSR to COO and vice-versa

# TODO: we can do similar for CSC conversions, but that requires the columns to be sorted

function CuSparseMatrixCSR(coo::CuSparseMatrixCOO{Tv}, ind::SparseChar='O') where {Tv}
    m,n = size(coo)
    csrRowPtr = CuVector{Cint}(undef, m+1)
    cusparseXcoo2csr(handle(), coo.rowInd, nnz(coo), m, csrRowPtr, ind)
    CuSparseMatrixCSR{Tv}(csrRowPtr, coo.colInd, nonzeros(coo), size(coo))
end

function CuSparseMatrixCOO(csr::CuSparseMatrixCSR{Tv}, ind::SparseChar='O') where {Tv}
    m,n = size(csr)
    cooRowInd = CuVector{Cint}(undef, nnz(csr))
    cusparseXcsr2coo(handle(), csr.rowPtr, nnz(csr), m, cooRowInd, ind)
    CuSparseMatrixCOO{Tv}(cooRowInd, csr.colVal, nonzeros(csr), size(csr), nnz(csr))
end

### CSC/BSR to COO and viceversa

CuSparseMatrixCSC(coo::CuSparseMatrixCOO) = CuSparseMatrixCSC(CuSparseMatrixCSR(coo)) # no direct conversion
CuSparseMatrixCOO(csc::CuSparseMatrixCSC) = CuSparseMatrixCOO(CuSparseMatrixCSR(csc)) # no direct conversion
CuSparseMatrixBSR(coo::CuSparseMatrixCOO, blockdim) = CuSparseMatrixBSR(CuSparseMatrixCSR(coo), blockdim) # no direct conversion
CuSparseMatrixCOO(bsr::CuSparseMatrixBSR) = CuSparseMatrixCOO(CuSparseMatrixCSR(bsr)) # no direct conversion


## sparse to dense, and vice-versa

for (cname,rname,elty) in ((:cusparseScsc2dense, :cusparseScsr2dense, :Float32),
                           (:cusparseDcsc2dense, :cusparseDcsr2dense, :Float64),
                           (:cusparseCcsc2dense, :cusparseCcsr2dense, :ComplexF32),
                           (:cusparseZcsc2dense, :cusparseZcsr2dense, :ComplexF64))
    @eval begin
        function CUDA.CuMatrix{$elty}(csr::CuSparseMatrixCSR{$elty}; ind::SparseChar='O')
            if version() >= v"11.3" # CUSPARSE version from CUDA release notes
                denseA = sparsetodense(csr, ind)
            else
                m,n = size(csr)
                denseA = CUDA.zeros($elty,m,n)
                cudesc = CuMatrixDescriptor('G', 'L', 'N', ind)
                lda = max(1,stride(denseA,2))
                $rname(handle(), m, n, cudesc, nonzeros(csr),
                       csr.rowPtr, csr.colVal, denseA, lda)
                return denseA
            end
            return denseA
        end
        function CUDA.CuMatrix{$elty}(csc::CuSparseMatrixCSC{$elty}; ind::SparseChar='O')
            if version() >= v"11.3" # CUSPARSE version from CUDA release notes
                denseA = sparsetodense(csc, ind)
            else
                m,n = size(csc)
                denseA = CUDA.zeros($elty,m,n)
                lda = max(1,stride(denseA,2))
                cudesc = CuMatrixDescriptor('G', 'L', 'N', ind)
                $cname(handle(), m, n, cudesc, nonzeros(csc),
                       rowvals(csc), csc.colPtr, denseA, lda)
            end
            return denseA
        end
    end
end

for (elty, welty) in ((:Float16, :Float32),
                      (:ComplexF16, :ComplexF32))
    @eval begin
        function CUDA.CuMatrix{$elty}(csr::CuSparseMatrixCSR{$elty}; ind::SparseChar='O')
            if version() >= v"11.3" # CUSPARSE version from CUDA release notes
                denseA = sparsetodense(csr, ind)
            else
                m,n = size(csr)
                denseA = CUDA.zeros($elty,m,n)
                wide_csr = CuSparseMatrixCSR(csr.rowPtr, csr.colVal, convert(CuVector{$welty}, nonzeros(csr)), size(csr))
                wide_dense = CuArray{$welty}(wide_csr)
                denseA = convert(CuArray{$elty}, wide_dense)
            end
            return denseA
        end
        function CUDA.CuMatrix{$elty}(csc::CuSparseMatrixCSC{$elty}; ind::SparseChar='O')
            if version() >= v"11.3" # CUSPARSE version from CUDA release notes
                denseA = sparsetodense(csc, ind)
            else
                m,n = size(csc)
                denseA = CUDA.zeros($elty,m,n)
                wide_csc = CuSparseMatrixCSC(csc.colPtr, csc.rowVal, convert(CuVector{$welty}, nonzeros(csc)), size(csc))
                wide_dense = CuArray{$welty}(wide_csc)
                denseA = convert(CuArray{$elty}, wide_dense)
            end
            return denseA
        end
    end
end

Base.copyto!(dest::Matrix{T}, src::AbstractCuSparseMatrix{T}) where T = copyto!(dest, CuMatrix{T}(src))

for (nname,cname,rname,elty) in ((:cusparseSnnz, :cusparseSdense2csc, :cusparseSdense2csr, :Float32),
                                 (:cusparseDnnz, :cusparseDdense2csc, :cusparseDdense2csr, :Float64),
                                 (:cusparseCnnz, :cusparseCdense2csc, :cusparseCdense2csr, :ComplexF32),
                                 (:cusparseZnnz, :cusparseZdense2csc, :cusparseZdense2csr, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR(A::CuMatrix{$elty}; ind::SparseChar='O', sorted::Bool=false)
            if !sorted && version() >= v"11.3" # CUSPARSE version from CUDA release notes
                return densetosparse(A, :csr, ind)
            else
                m,n = size(A)
                lda = max(1, stride(A,2))
                cudesc = CuMatrixDescriptor('G',
                                            'L',
                                            'N', ind)
                nnzRowCol = CUDA.zeros(Cint, m)
                nnzTotal = Ref{Cint}(1)
                $nname(handle(),
                       'R', m, n, cudesc, A, lda, nnzRowCol,
                       nnzTotal)
                nzVal = CUDA.zeros($elty,nnzTotal[])

                rowPtr = CUDA.zeros(Cint,m+1)
                colInd = CUDA.zeros(Cint,nnzTotal[])
                $rname(handle(), m, n, cudesc, A,
                        lda, nnzRowCol, nzVal, rowPtr, colInd)
                return CuSparseMatrixCSR(rowPtr,colInd,nzVal,size(A))
            end
        end

        function CuSparseMatrixCSC(A::CuMatrix{$elty}; ind::SparseChar='O', sorted::Bool=false)
            if !sorted && version() >= v"11.3" # CUSPARSE version from CUDA release notes
                return densetosparse(A, :csc, ind)
            else
                m,n = size(A)
                lda = max(1, stride(A,2))
                cudesc = CuMatrixDescriptor('G',
                                            'L',
                                            'N', ind)
                nnzRowCol = CUDA.zeros(Cint, n)
                nnzTotal = Ref{Cint}(1)
                $nname(handle(),
                       'C', m, n, cudesc, A, lda, nnzRowCol,
                       nnzTotal)
                nzVal = CUDA.zeros($elty,nnzTotal[])

                colPtr = CUDA.zeros(Cint,n+1)
                rowInd = CUDA.zeros(Cint,nnzTotal[])
                $cname(handle(), m, n, cudesc, A,
                        lda, nnzRowCol, nzVal, rowInd, colPtr)
                return CuSparseMatrixCSC(colPtr,rowInd,nzVal,size(A))
            end
        end
    end
end

for (elty, welty) in ((:Float16, :Float32),
                      (:ComplexF16, :ComplexF32))
    @eval begin
        function CuSparseMatrixCSR(A::CuMatrix{$elty}; ind::SparseChar='O', sorted::Bool=false)
            if !sorted && version() >= v"11.3" # CUSPARSE version from CUDA release notes
                return densetosparse(A, :csr, ind)
            else
                wide_csr = CuSparseMatrixCSR(convert(CuMatrix{$welty}, A))
                return CuSparseMatrixCSR(wide_csr.rowPtr, wide_csr.colVal, convert(CuVector{$elty}, nonzeros(wide_csr)), size(wide_csr))
            end
        end
        function CuSparseMatrixCSC(A::CuMatrix{$elty}; ind::SparseChar='O', sorted::Bool=false)
            if !sorted && version() >= v"11.3" # CUSPARSE version from CUDA release notes
                return densetosparse(A, :csc, ind)
            else
                wide_csc = CuSparseMatrixCSC(convert(CuMatrix{$welty}, A))
                return CuSparseMatrixCSC(wide_csc.colPtr, wide_csc.rowVal, convert(CuVector{$elty}, nonzeros(wide_csc)), size(wide_csc))
            end
        end
    end
end

function CUDA.CuMatrix{T}(bsr::CuSparseMatrixBSR{T}; inda::SparseChar='O',
                          indc::SparseChar='O') where {T}
    CuMatrix{T}(CuSparseMatrixCSR{T}(bsr; inda, indc))
end

function CuSparseMatrixBSR(A::CuMatrix, blockDim::Integer=gcd(size(A)...); ind::SparseChar='O')
    m,n = size(A)
    # csr.colVal should be sorted if we want to use "csr2bsr" routines.
    csr = CuSparseMatrixCSR(A; ind, sorted=true)
    CuSparseMatrixBSR(csr, blockDim)
end

function CUDA.CuMatrix{T}(coo::CuSparseMatrixCOO{T}; ind::SparseChar='O') where {T}
    sparsetodense(coo, ind)
end

function CuSparseMatrixCOO(A::CuMatrix; ind::SparseChar='O')
    densetosparse(A, :coo, ind)
end
