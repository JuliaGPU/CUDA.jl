# conversion routines between different sparse and dense storage formats

SparseArrays.sparse(::DenseCuArray, args...) = error("CUSPARSE supports multiple sparse formats, use specific constructors instead (e.g. CuSparseMatrixCSC)")


## CSR to CSC

# by flipping rows and columns, we can use that to get CSC to CSR too

for (fname,elty) in ((:cusparseScsr2csc, :Float32),
                     (:cusparseDcsr2csc, :Float64),
                     (:cusparseCcsr2csc, :ComplexF32),
                     (:cusparseZcsr2csc, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty}; inda::SparseChar='O')
            m,n = csr.dims
            colPtr = CUDA.zeros(Cint, n+1)
            rowVal = CUDA.zeros(Cint, nnz(csr))
            nzVal = CUDA.zeros($elty, nnz(csr))
            if version() >= v"10.2"
                # TODO: algorithm configuratibility?
                @workspace size=@argout(
                        cusparseCsr2cscEx2_bufferSize(handle(), m, n, nnz(csr), nonzeros(csr),
                            csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, out(Ref{Csize_t}(1)))
                    )[] buffer->begin
                        cusparseCsr2cscEx2(handle(), m, n, nnz(csr), nonzeros(csr),
                            csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, buffer)
                    end
            else
                $fname(handle(), m, n, nnz(csr), nonzeros(csr),
                    csr.rowPtr, csr.colVal, nzVal, rowVal,
                    colPtr, CUSPARSE_ACTION_NUMERIC, inda)
            end
            CuSparseMatrixCSC(colPtr,rowVal,nzVal,csr.dims)
        end

        function CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty}; inda::SparseChar='O')
            m,n    = csc.dims
            rowPtr = CUDA.zeros(Cint,m+1)
            colVal = CUDA.zeros(Cint,nnz(csc))
            nzVal  = CUDA.zeros($elty,nnz(csc))
            if version() >= v"10.2"
                # TODO: algorithm configuratibility?
                @workspace size=@argout(
                        cusparseCsr2cscEx2_bufferSize(handle(), n, m, nnz(csc), nonzeros(csc),
                            csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, out(Ref{Csize_t}(1)))
                    )[] buffer->begin
                        cusparseCsr2cscEx2(handle(), n, m, nnz(csc), nonzeros(csc),
                            csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, buffer)
                    end
            else
                $fname(handle(), n, m, nnz(csc), nonzeros(csc),
                    csc.colPtr, rowvals(csc), nzVal, colVal,
                    rowPtr, CUSPARSE_ACTION_NUMERIC, inda)
            end
            CuSparseMatrixCSR(rowPtr,colVal,nzVal,csc.dims)
        end
    end
end


## CSR to BSR and vice-versa

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :ComplexF32),
                     (:cusparseZcsr2bsr, :ComplexF64))
    @eval begin
        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty}, blockDim::Integer;
                                          dir::SparseChar='R', inda::SparseChar='O',
                                          indc::SparseChar='O')
            m,n = csr.dims
            nnz_ref = Ref{Cint}(1)
            mb = div((m + blockDim - 1),blockDim)
            nb = div((n + blockDim - 1),blockDim)
            bsrRowPtr = CUDA.zeros(Cint,mb + 1)
            cudesca = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, inda)
            cudescc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, indc)
            cusparseXcsr2bsrNnz(handle(), dir, m, n, cudesca, csr.rowPtr,
                                csr.colVal, blockDim, cudescc, bsrRowPtr, nnz_ref)
            bsrNzVal = CUDA.zeros($elty, nnz_ref[] * blockDim * blockDim )
            bsrColInd = CUDA.zeros(Cint, nnz_ref[])
            $fname(handle(), dir, m, n,
                   cudesca, nonzeros(csr), csr.rowPtr, csr.colVal,
                   blockDim, cudescc, bsrNzVal, bsrRowPtr,
                   bsrColInd)
            CuSparseMatrixBSR{$elty}(bsrRowPtr, bsrColInd, bsrNzVal, csr.dims, blockDim, dir, nnz_ref[])
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :ComplexF32),
                     (:cusparseZbsr2csr, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty};
                                          inda::SparseChar='O', indc::SparseChar='O')
            m,n = bsr.dims
            mb = div(m,bsr.blockDim)
            nb = div(n,bsr.blockDim)
            nnzVal = nnz(bsr) * bsr.blockDim * bsr.blockDim
            cudesca = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, inda)
            cudescc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, indc)
            csrRowPtr = CUDA.zeros(Cint, m + 1)
            csrColInd = CUDA.zeros(Cint, nnzVal)
            csrNzVal  = CUDA.zeros($elty, nnzVal)
            $fname(handle(), bsr.dir, mb, nb,
                   cudesca, nonzeros(bsr), bsr.rowPtr, bsr.colVal,
                   bsr.blockDim, cudescc, csrNzVal, csrRowPtr,
                   csrColInd)
            CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, bsr.dims)
        end
    end
end

## CSR to COO and vice-versa

function CuSparseMatrixCSR(coo::CuSparseMatrixCOO{Tv}, ind::SparseChar='O') where {Tv}
    m,n = coo.dims
    nnz = coo.nnz
    csrRowPtr = CUDA.zeros(Cint, nnz)
    cusparseXcoo2csr(handle(), coo.rowInd, nnz, m, csrRowPtr, ind)
    CuSparseMatrixCSR{Tv}(csrRowPtr, coo.colInd, coo.nzVal, coo.dims)
end

function CuSparseMatrixCOO(csr::CuSparseMatrixCSR{Tv}, ind::SparseChar='O') where {Tv}
    m,n = csr.dims
    nnz = csr.nnz
    cooRowInd = CUDA.zeros(Cint, nnz)
    cusparseXcsr2coo(handle(), csr.rowPtr, nnz, m, cooRowInd, ind)
    CuSparseMatrixCOO{Tv}(cooRowInd, csr.colVal, csr.nzVal, csr.dims, nnz)
end

## sparse to dense, and vice-versa

for (cname,rname,elty) in ((:cusparseScsc2dense, :cusparseScsr2dense, :Float32),
                           (:cusparseDcsc2dense, :cusparseDcsr2dense, :Float64),
                           (:cusparseCcsc2dense, :cusparseCcsr2dense, :ComplexF32),
                           (:cusparseZcsc2dense, :cusparseZcsr2dense, :ComplexF64))
    @eval begin
        function CUDA.CuMatrix{$elty}(csr::CuSparseMatrixCSR{$elty}; ind::SparseChar='O')
            m,n = csr.dims
            denseA = CUDA.zeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
            $rname(handle(), m, n, cudesc, nonzeros(csr),
                   csr.rowPtr, csr.colVal, denseA, lda)
            denseA
        end
        function CUDA.CuMatrix{$elty}(csc::CuSparseMatrixCSC{$elty}; ind::SparseChar='O')
            m,n = csc.dims
            denseA = CUDA.zeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
            $cname(handle(), m, n, cudesc, nonzeros(csc),
                   rowvals(csc), csc.colPtr, denseA, lda)
            denseA
        end
    end
end

for (nname,cname,rname,elty) in ((:cusparseSnnz, :cusparseSdense2csc, :cusparseSdense2csr, :Float32),
                                 (:cusparseDnnz, :cusparseDdense2csc, :cusparseDdense2csr, :Float64),
                                 (:cusparseCnnz, :cusparseCdense2csc, :cusparseCdense2csr, :ComplexF32),
                                 (:cusparseZnnz, :cusparseZdense2csc, :cusparseZdense2csr, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR(A::CuMatrix{$elty}; ind::SparseChar='O')
            m,n = size(A)
            lda = max(1, stride(A,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL,
                                        CUSPARSE_FILL_MODE_LOWER,
                                        CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
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

        function CuSparseMatrixCSC(A::CuMatrix{$elty}; ind::SparseChar='O')
            m,n = size(A)
            lda = max(1, stride(A,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL,
                                        CUSPARSE_FILL_MODE_LOWER,
                                        CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
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

function CUDA.CuMatrix{T}(bsr::CuSparseMatrixBSR{T}; inda::SparseChar='O',
                          indc::SparseChar='O') where {T}
    CuMatrix{T}(CuSparseMatrixCSR{T}(bsr; inda, indc))
end

function CuSparseMatrixBSR(A::CuMatrix; ind::SparseChar='O')
    m,n = size(A)   # TODO: always let the user choose, or provide defaults for other methods too
    CuSparseMatrixBSR(CuSparseMatrixCSR(A; ind), gcd(m,n))
end
