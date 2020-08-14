# conversion routines between different sparse and dense storage formats

export switch2csr, switch2csc, switch2bsr

for (fname,elty) in ((:cusparseScsr2csc, :Float32),
                     (:cusparseDcsr2csc, :Float64),
                     (:cusparseCcsr2csc, :ComplexF32),
                     (:cusparseZcsr2csc, :ComplexF64))
    @eval begin
        function switch2csc(csr::CuSparseMatrixCSR{$elty},inda::SparseChar='O')
            m,n = csr.dims
            colPtr = CUDA.zeros(Cint, n+1)
            rowVal = CUDA.zeros(Cint, csr.nnz)
            nzVal = CUDA.zeros($elty, csr.nnz)
            if version() >= v"10.2"
                # TODO: algorithm configuratibility?
                @workspace size=@argout(
                        cusparseCsr2cscEx2_bufferSize(handle(), m, n, csr.nnz, csr.nzVal,
                            csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, out(Ref{Csize_t}(1)))
                    )[] buffer->begin
                        cusparseCsr2cscEx2(handle(), m, n, csr.nnz, csr.nzVal,
                            csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, buffer)
                    end
            else
                $fname(handle(), m, n, csr.nnz, csr.nzVal,
                    csr.rowPtr, csr.colVal, nzVal, rowVal,
                    colPtr, CUSPARSE_ACTION_NUMERIC, inda)
            end
            csc = CuSparseMatrixCSC(colPtr,rowVal,nzVal,csr.nnz,csr.dims)
            csc
        end
        function switch2csr(csc::CuSparseMatrixCSC{$elty},inda::SparseChar='O')
            m,n    = csc.dims
            rowPtr = CUDA.zeros(Cint,m+1)
            colVal = CUDA.zeros(Cint,csc.nnz)
            nzVal  = CUDA.zeros($elty,csc.nnz)
            if version() >= v"10.2"
                # TODO: algorithm configuratibility?
                @workspace size=@argout(
                        cusparseCsr2cscEx2_bufferSize(handle(), n, m, csc.nnz, csc.nzVal,
                            csc.colPtr, csc.rowVal, nzVal, rowPtr, colVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, out(Ref{Csize_t}(1)))
                    )[] buffer->begin
                        cusparseCsr2cscEx2(handle(), n, m, csc.nnz, csc.nzVal,
                            csc.colPtr, csc.rowVal, nzVal, rowPtr, colVal,
                            $elty, CUSPARSE_ACTION_NUMERIC, inda,
                            CUSPARSE_CSR2CSC_ALG1, buffer)
                    end
            else
                $fname(handle(), n, m, csc.nnz, csc.nzVal,
                    csc.colPtr, csc.rowVal, nzVal, colVal,
                    rowPtr, CUSPARSE_ACTION_NUMERIC, inda)
            end
            csr = CuSparseMatrixCSR(rowPtr,colVal,nzVal,csc.nnz,csc.dims)
            csr
        end
    end
end

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :ComplexF32),
                     (:cusparseZcsr2bsr, :ComplexF64))
    @eval begin
        function switch2bsr(csr::CuSparseMatrixCSR{$elty},
                            blockDim::Cint,
                            dir::SparseChar='R',
                            inda::SparseChar='O',
                            indc::SparseChar='O')
            m,n = csr.dims
            nnz = Ref{Cint}(1)
            mb = div((m + blockDim - 1),blockDim)
            nb = div((n + blockDim - 1),blockDim)
            bsrRowPtr = CUDA.zeros(Cint,mb + 1)
            cudesca = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, inda)
            cudescc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, indc)
            cusparseXcsr2bsrNnz(handle(), dir, m, n, cudesca, csr.rowPtr,
                                csr.colVal, blockDim, cudescc, bsrRowPtr, nnz)
            bsrNzVal = CUDA.zeros($elty, nnz[] * blockDim * blockDim )
            bsrColInd = CUDA.zeros(Cint, nnz[])
            $fname(handle(), dir, m, n,
                   cudesca, csr.nzVal, csr.rowPtr, csr.colVal,
                   blockDim, cudescc, bsrNzVal, bsrRowPtr,
                   bsrColInd)
            CuSparseMatrixBSR{$elty}(bsrRowPtr, bsrColInd, bsrNzVal, csr.dims, blockDim, dir, nnz[])
        end
        function switch2bsr(csc::CuSparseMatrixCSC{$elty},
                            blockDim::Cint,
                            dir::SparseChar='R',
                            inda::SparseChar='O',
                            indc::SparseChar='O')
                switch2bsr(switch2csr(csc),blockDim,dir,inda,indc)
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :ComplexF32),
                     (:cusparseZbsr2csr, :ComplexF64))
    @eval begin
        function switch2csr(bsr::CuSparseMatrixBSR{$elty},
                            inda::SparseChar='O',
                            indc::SparseChar='O')
            m,n = bsr.dims
            mb = div(m,bsr.blockDim)
            nb = div(n,bsr.blockDim)
            nnz = bsr.nnz * bsr.blockDim * bsr.blockDim
            cudesca = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, inda)
            cudescc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, indc)
            csrRowPtr = CUDA.zeros(Cint, m + 1)
            csrColInd = CUDA.zeros(Cint, nnz)
            csrNzVal  = CUDA.zeros($elty, nnz)
            $fname(handle(), bsr.dir, mb, nb,
                   cudesca, bsr.nzVal, bsr.rowPtr, bsr.colVal,
                   bsr.blockDim, cudescc, csrNzVal, csrRowPtr,
                   csrColInd)
            CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, convert(Cint,nnz), bsr.dims)
        end
        function switch2csc(bsr::CuSparseMatrixBSR{$elty},
                            inda::SparseChar='O',
                            indc::SparseChar='O')
            switch2csc(switch2csr(bsr,inda,indc))
        end
    end
end

for (cname,rname,elty) in ((:cusparseScsc2dense, :cusparseScsr2dense, :Float32),
                           (:cusparseDcsc2dense, :cusparseDcsr2dense, :Float64),
                           (:cusparseCcsc2dense, :cusparseCcsr2dense, :ComplexF32),
                           (:cusparseZcsc2dense, :cusparseZcsr2dense, :ComplexF64))
    @eval begin
        function Base.Array(csr::CuSparseMatrixCSR{$elty},ind::SparseChar='O')
            m,n = csr.dims
            denseA = CUDA.zeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
            $rname(handle(), m, n, cudesc, csr.nzVal,
                   csr.rowPtr, csr.colVal, denseA, lda)
            denseA
        end
        function Base.Array(csc::CuSparseMatrixCSC{$elty},ind::SparseChar='O')
            m,n = csc.dims
            denseA = CUDA.zeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
            $cname(handle(), m, n, cudesc, csc.nzVal,
                   csc.rowVal, csc.colPtr, denseA, lda)
            denseA
        end
        function Base.Array(bsr::CuSparseMatrixBSR{$elty},ind::SparseChar='O')
            Array(switch2csr(bsr,ind))
        end
    end
end

for (nname,cname,rname,elty) in ((:cusparseSnnz, :cusparseSdense2csc, :cusparseSdense2csr, :Float32),
                                 (:cusparseDnnz, :cusparseDdense2csc, :cusparseDdense2csr, :Float64),
                                 (:cusparseCnnz, :cusparseCdense2csc, :cusparseCdense2csr, :ComplexF32),
                                 (:cusparseZnnz, :cusparseZdense2csc, :cusparseZdense2csr, :ComplexF64))
    @eval begin
        function sparse(A::CuMatrix{$elty},fmt::SparseChar='R',ind::SparseChar='O')
            dir = 'R'
            if fmt == 'C'
                dir = fmt
            end
            m,n = size(A)
            lda = max(1,stride(A,2))
            cudesc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, ind)
            nnzRowCol = CUDA.zeros(Cint, fmt == 'R' ? m : n)
            nnzTotal = Ref{Cint}(1)
            $nname(handle(),
                   dir, m, n, cudesc, A, lda, nnzRowCol,
                   nnzTotal)
            nzVal = CUDA.zeros($elty,nnzTotal[])
            if(fmt == 'R')
                rowPtr = CUDA.zeros(Cint,m+1)
                colInd = CUDA.zeros(Cint,nnzTotal[])
                $rname(handle(), m, n, cudesc, A,
                       lda, nnzRowCol, nzVal, rowPtr, colInd)
                return CuSparseMatrixCSR(rowPtr,colInd,nzVal,nnzTotal[],size(A))
            end
            if(fmt == 'C')
                colPtr = CUDA.zeros(Cint,n+1)
                rowInd = CUDA.zeros(Cint,nnzTotal[])
                $cname(handle(), m, n, cudesc, A,
                       lda, nnzRowCol, nzVal, rowInd, colPtr)
                return CuSparseMatrixCSC(colPtr,rowInd,nzVal,nnzTotal[],size(A))
            end
            if(fmt == 'B')
                return switch2bsr(sparse(A,'R',ind),convert(Cint,gcd(m,n)))
            end
        end
    end
end

"""
    switch2csr(csr::CuSparseMatrixCSR, inda::SparseChar='O')

Convert a `CuSparseMatrixCSR` to the compressed sparse column format.
"""
switch2csc(csr::CuSparseMatrixCSR, inda::SparseChar='O')

"""
    switch2csr(csc::CuSparseMatrixCSC, inda::SparseChar='O')

Convert a `CuSparseMatrixCSC` to the compressed sparse row format.
"""
switch2csr(csc::CuSparseMatrixCSC, inda::SparseChar='O')

"""
    switch2bsr(csr::CuSparseMatrixCSR, blockDim::Cint, dir::SparseChar='R', inda::SparseChar='O', indc::SparseChar='O')

Convert a `CuSparseMatrixCSR` to the compressed block sparse row format. `blockDim` sets the
block dimension of the compressed sparse blocks and `indc` determines whether the new matrix
will be one- or zero-indexed.
"""
switch2bsr(csr::CuSparseMatrixCSR, blockDim::Cint, dir::SparseChar='R', inda::SparseChar='O', indc::SparseChar='O')
