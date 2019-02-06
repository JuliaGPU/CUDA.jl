# utility functions for the CUSPARSE wrappers
#
# TODO: move raw ccall wrappers to libcusparse.jl

"""
convert `SparseChar` {`N`,`T`,`C`} to `cusparseOperation_t`
"""
function cusparseop(trans::SparseChar)
    if trans == 'N'
        return CUSPARSE_OPERATION_NON_TRANSPOSE
    end
    if trans == 'T'
        return CUSPARSE_OPERATION_TRANSPOSE
    end
    if trans == 'C'
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    end
    throw(ArgumentError("unknown cusparse operation $trans"))
end

"""
convert `SparseChar` {`G`,`S`,`H`,`T`} to `cusparseMatrixType_t`
"""
function cusparsetype(mattype::SparseChar)
    if mattype == 'G'
        return CUSPARSE_MATRIX_TYPE_GENERAL
    end
    if mattype == 'T'
        return CUSPARSE_MATRIX_TYPE_TRIANGULAR
    end
    if mattype == 'S'
        return CUSPARSE_MATRIX_TYPE_SYMMETRIC
    end
    if mattype == 'H'
        return CUSPARSE_MATRIX_TYPE_HERMITIAN
    end
    throw(ArgumentError("unknown cusparse matrix type $mattype"))
end

"""
convert `SparseChar` {`U`,`L`} to `cusparseFillMode_t`
"""
function cusparsefill(uplo::SparseChar)
    if uplo == 'U'
        return CUSPARSE_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUSPARSE_FILL_MODE_LOWER
    end
    throw(ArgumentError("unknown cusparse fill mode $uplo"))
end

"""
convert `SparseChar` {`U`,`N`} to `cusparseDiagType_t`
"""
function cusparsediag(diag::SparseChar)
    if diag == 'U'
        return CUSPARSE_DIAG_TYPE_UNIT
    end
    if diag == 'N'
        return CUSPARSE_DIAG_TYPE_NON_UNIT
    end
    throw(ArgumentError("unknown cusparse diag mode $diag"))
end

"""
convert `SparseChar` {`Z`,`O`} to `cusparseIndexBase_t`
"""
function cusparseindex(index::SparseChar)
    if index == 'Z'
        return CUSPARSE_INDEX_BASE_ZERO
    end
    if index == 'O'
        return CUSPARSE_INDEX_BASE_ONE
    end
    throw(ArgumentError("unknown cusparse index base"))
end

"""
convert `SparseChar` {`R`,`C`} to `cusparseDirection_t`
"""
function cusparsedir(dir::SparseChar)
    if dir == 'R'
        return CUSPARSE_DIRECTION_ROW
    end
    if dir == 'C'
        return CUSPARSE_DIRECTION_COL
    end
    throw(ArgumentError("unknown cusparse direction $dir"))
end

"""
check that the dimensions of matrix `X` and vector `Y` make sense for a multiplication
"""
function chkmvdims(X, n, Y, m)
    if length(X) != n
        throw(DimensionMismatch("X must have length $n, but has length $(length(X))"))
    elseif length(Y) != m
        throw(DimensionMismatch("Y must have length $m, but has length $(length(Y))"))
    end
end

"""
check that the dimensions of matrices `X` and `Y` make sense for a multiplication
"""
function chkmmdims( B, C, k, l, m, n )
    if size(B) != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C) != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end

"""
form a `cusparseMatDescr_t` from a `CuSparseMatrix`
"""
function getDescr( A::CuSparseMatrix, index::SparseChar )
    cuind = cusparseindex(index)
    typ   = CUSPARSE_MATRIX_TYPE_GENERAL
    fill  = CUSPARSE_FILL_MODE_LOWER
    if ishermitian(A)
        typ  = CUSPARSE_MATRIX_TYPE_HERMITIAN
        fill = cusparsefill(A.uplo)
    elseif issymmetric(A)
        typ  = CUSPARSE_MATRIX_TYPE_SYMMETRIC
        fill = cusparsefill(A.uplo)
    end
    cudesc = cusparseMatDescr_t(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

function getDescr( A::Symmetric, index::SparseChar )
    cuind = cusparseindex(index)
    typ  = CUSPARSE_MATRIX_TYPE_SYMMETRIC
    fill = cusparsefill(A.uplo)
    cudesc = cusparseMatDescr_t(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

function getDescr( A::Hermitian, index::SparseChar )
    cuind = cusparseindex(index)
    typ  = CUSPARSE_MATRIX_TYPE_HERMITIAN
    fill = cusparsefill(A.uplo)
    cudesc = cusparseMatDescr_t(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

# type conversion
for (fname,elty) in ((:cusparseScsr2csc, :Float32),
                     (:cusparseDcsr2csc, :Float64),
                     (:cusparseCcsr2csc, :ComplexF32),
                     (:cusparseZcsr2csc, :ComplexF64))
    @eval begin
        function switch2csc(csr::CuSparseMatrixCSR{$elty},inda::SparseChar='O')
            cuind = cusparseindex(inda)
            m,n = csr.dims
            colPtr = cuzeros(Cint, n+1)
            rowVal = cuzeros(Cint, csr.nnz)
            nzVal = cuzeros($elty, csr.nnz)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                               handle(), m, n, csr.nnz, csr.nzVal,
                               csr.rowPtr, csr.colVal, nzVal, rowVal,
                               colPtr, CUSPARSE_ACTION_NUMERIC, cuind)
            csc = CuSparseMatrixCSC(colPtr,rowVal,nzVal,csr.nnz,csr.dims)
            csc
        end
        function switch2csr(csc::CuSparseMatrixCSC{$elty},inda::SparseChar='O')
            cuind = cusparseindex(inda)
            m,n = csc.dims
            rowPtr = cuzeros(Cint,m+1)
            colVal = cuzeros(Cint,csc.nnz)
            nzVal = cuzeros($elty,csc.nnz)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                               handle(), n, m, csc.nnz, csc.nzVal,
                               csc.colPtr, csc.rowVal, nzVal, colVal,
                               rowPtr, CUSPARSE_ACTION_NUMERIC, cuind)
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
            cudir = cusparsedir(dir)
            cuinda = cusparseindex(inda)
            cuindc = cusparseindex(indc)
            m,n = csr.dims
            nnz = Ref{Cint}(1)
            mb = div((m + blockDim - 1),blockDim)
            nb = div((n + blockDim - 1),blockDim)
            bsrRowPtr = cuzeros(Cint,mb + 1)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            @check ccall((:cusparseXcsr2bsrNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{Cint},
                               CuPtr{Cint}, Cint, Ptr{cusparseMatDescr_t},
                               CuPtr{Cint}, Ptr{Cint}),
                              handle(), cudir, m, n, Ref(cudesca), csr.rowPtr,
                              csr.colVal, blockDim, Ref(cudescc), bsrRowPtr, nnz)
            bsrNzVal = cuzeros($elty, nnz[] * blockDim * blockDim )
            bsrColInd = cuzeros(Cint, nnz[])
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}), handle(), cudir, m, n,
                               Ref(cudesca), csr.nzVal, csr.rowPtr, csr.colVal,
                               blockDim, Ref(cudescc), bsrNzVal, bsrRowPtr,
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
            cudir = cusparsedir(bsr.dir)
            cuinda = cusparseindex(inda)
            cuindc = cusparseindex(indc)
            m,n = bsr.dims
            mb = div(m,bsr.blockDim)
            nb = div(n,bsr.blockDim)
            nnz = bsr.nnz * bsr.blockDim * bsr.blockDim
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            csrRowPtr = cuzeros(Cint, m + 1)
            csrColInd = cuzeros(Cint, nnz)
            csrNzVal  = cuzeros($elty, nnz)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}), handle(), cudir, mb, nb,
                               Ref(cudesca), bsr.nzVal, bsr.rowPtr, bsr.colVal,
                               bsr.blockDim, Ref(cudescc), csrNzVal, csrRowPtr,
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
            cuind = cusparseindex(ind)
            m,n = csr.dims
            denseA = cuzeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            @check ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty}, Cint),
                               handle(), m, n, Ref(cudesc), csr.nzVal,
                               csr.rowPtr, csr.colVal, denseA, lda)
            denseA
        end
        function Base.Array(csc::CuSparseMatrixCSC{$elty},ind::SparseChar='O')
            cuind = cusparseindex(ind)
            m,n = csc.dims
            denseA = cuzeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            @check ccall(($(string(cname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty}, Cint),
                               handle(), m, n, Ref(cudesc), csc.nzVal,
                               csc.rowVal, csc.colPtr, denseA, lda)
            denseA
        end
        function Base.Array(hyb::CuSparseMatrixHYB{$elty},ind::SparseChar='O')
            Array(switch2csr(hyb,ind))
        end
        function Base.Array(bsr::CuSparseMatrixBSR{$elty},ind::SparseChar='O')
            Array(switch2csr(bsr,ind))
        end
    end
end

for (nname,cname,rname,hname,elty) in ((:cusparseSnnz, :cusparseSdense2csc, :cusparseSdense2csr, :cusparseSdense2hyb, :Float32),
                                       (:cusparseDnnz, :cusparseDdense2csc, :cusparseDdense2csr, :cusparseDdense2hyb, :Float64),
                                       (:cusparseCnnz, :cusparseCdense2csc, :cusparseCdense2csr, :cusparseCdense2hyb, :ComplexF32),
                                       (:cusparseZnnz, :cusparseZdense2csc, :cusparseZdense2csr, :cusparseZdense2hyb, :ComplexF64))
    @eval begin
        function sparse(A::CuMatrix{$elty},fmt::SparseChar='R',ind::SparseChar='O')
            cuind = cusparseindex(ind)
            cudir = cusparsedir('R')
            if( fmt == 'C' )
                cudir = cusparsedir(fmt)
            end
            m,n = size(A)
            lda = max(1,stride(A,2))
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            nnzRowCol = cuzeros(Cint, fmt == 'R' ? m : n)
            nnzTotal = Ref{Cint}(1)
            @check ccall(($(string(nname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               Cint, Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               Cint, CuPtr{Cint}, Ptr{Cint}), handle(),
                               cudir, m, n, Ref(cudesc), A, lda, nnzRowCol,
                               nnzTotal)
            nzVal = cuzeros($elty,nnzTotal[])
            if(fmt == 'R')
                rowPtr = cuzeros(Cint,m+1)
                colInd = cuzeros(Cint,nnzTotal[])
                @check ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                                  (cusparseHandle_t, Cint, Cint,
                                   Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                                   Cint, CuPtr{Cint}, CuPtr{$elty}, CuPtr{Cint},
                                   CuPtr{Cint}), handle(), m, n, Ref(cudesc), A,
                                   lda, nnzRowCol, nzVal, rowPtr, colInd)
                return CuSparseMatrixCSR(rowPtr,colInd,nzVal,nnzTotal[],size(A))
            end
            if(fmt == 'C')
                colPtr = cuzeros(Cint,n+1)
                rowInd = cuzeros(Cint,nnzTotal[])
                @check ccall(($(string(cname)),libcusparse), cusparseStatus_t,
                                  (cusparseHandle_t, Cint, Cint,
                                   Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                                   Cint, CuPtr{Cint}, CuPtr{$elty}, CuPtr{Cint},
                                   CuPtr{Cint}), handle(), m, n, Ref(cudesc), A,
                                   lda, nnzRowCol, nzVal, rowInd, colPtr)
                return CuSparseMatrixCSC(colPtr,rowInd,nzVal,nnzTotal[],size(A))
            end
            if(fmt == 'B')
                return switch2bsr(sparse(A,'R',ind),convert(Cint,gcd(m,n)))
            end
            if(fmt == 'H')
                hyb = cusparseHybMat_t[0]
                @check ccall((:cusparseCreateHybMat,libcusparse), cusparseStatus_t,
                                  (Ptr{cusparseHybMat_t},), hyb)
                @check ccall(($(string(hname)),libcusparse), cusparseStatus_t,
                                  (cusparseHandle_t, Cint, Cint,
                                   Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                                   Cint, CuPtr{Cint}, cusparseHybMat_t,
                                   Cint, cusparseHybPartition_t),
                                  handle(), m, n, Ref(cudesc), A, lda, nnzRowCol,
                                  hyb[1], 0, CUSPARSE_HYB_PARTITION_AUTO)
                return CuSparseMatrixHYB{$elty}(hyb[1],size(A),nnzTotal[])
            end
        end
    end
end

"""
    switch2hyb(csr::CuSparseMatrixCSR, inda::SparseChar='O')

Convert a `CuSparseMatrixCSR` to the hybrid CUDA storage format.
"""
switch2hyb(csr::CuSparseMatrixCSR, inda::SparseChar='O')

"""
    switch2hyb(csc::CuSparseMatrixCSC, inda::SparseChar='O')

Convert a `CuSparseMatrixCSC` to the hybrid CUDA storage format.
"""
switch2hyb(csc::CuSparseMatrixCSC, inda::SparseChar='O')

for (rname,cname,elty) in ((:cusparseScsr2hyb, :cusparseScsc2hyb, :Float32),
                           (:cusparseDcsr2hyb, :cusparseDcsc2hyb, :Float64),
                           (:cusparseCcsr2hyb, :cusparseCcsc2hyb, :ComplexF32),
                           (:cusparseZcsr2hyb, :cusparseZcsc2hyb, :ComplexF64))
    @eval begin
        function switch2hyb(csr::CuSparseMatrixCSR{$elty},
                            inda::SparseChar='O')
            cuinda = cusparseindex(inda)
            m,n = csr.dims
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            hyb = cusparseHybMat_t[0]
            @check ccall((:cusparseCreateHybMat,libcusparse), cusparseStatus_t,
                              (Ptr{cusparseHybMat_t},), hyb)
            @check ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t,
                               Cint, cusparseHybPartition_t), handle(),
                               m, n, Ref(cudesca), csr.nzVal, csr.rowPtr, csr.colVal,
                               hyb[1], 0, CUSPARSE_HYB_PARTITION_AUTO)
            CuSparseMatrixHYB{$elty}(hyb[1], csr.dims, csr.nnz)
        end
        function switch2hyb(csc::CuSparseMatrixCSC{$elty},
                            inda::SparseChar='O')
            switch2hyb(switch2csr(csc,inda),inda)
        end
    end
end

for (rname,cname,elty) in ((:cusparseShyb2csr, :cusparseShyb2csc, :Float32),
                           (:cusparseDhyb2csr, :cusparseDhyb2csc, :Float64),
                           (:cusparseChyb2csr, :cusparseChyb2csc, :ComplexF32),
                           (:cusparseZhyb2csr, :cusparseZhyb2csc, :ComplexF64))
    @eval begin
        function switch2csr(hyb::CuSparseMatrixHYB{$elty},
                            inda::SparseChar='O')
            cuinda = cusparseindex(inda)
            m,n = hyb.dims
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            csrRowPtr = cuzeros(Cint, m + 1)
            csrColInd = cuzeros(Cint, hyb.nnz)
            csrNzVal = cuzeros($elty, hyb.nnz)
            @check ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}), handle(), Ref(cudesca),
                               hyb.Mat, csrNzVal, csrRowPtr, csrColInd)
            CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, hyb.nnz, hyb.dims)
        end
        function switch2csc(hyb::CuSparseMatrixHYB{$elty},
                            inda::SparseChar='O')
            switch2csc(switch2csr(hyb,inda),inda)
        end
    end
end

"""
    switch2csr(csr::CuSparseMatrixCSR, inda::SparseChar='O')

Convert a `CuSparseMatrixCSR` to the compressed sparse column format.
"""
function switch2csc(csr::CuSparseMatrixCSR, inda::SparseChar='O') end

"""
    switch2csr(csc::CuSparseMatrixCSC, inda::SparseChar='O')

Convert a `CuSparseMatrixCSC` to the compressed sparse row format.
"""
function switch2csr(csc::CuSparseMatrixCSC, inda::SparseChar='O') end

"""
    switch2bsr(csr::CuSparseMatrixCSR, blockDim::Cint, dir::SparseChar='R', inda::SparseChar='O', indc::SparseChar='O')

Convert a `CuSparseMatrixCSR` to the compressed block sparse row format. `blockDim` sets the block dimension of the compressed sparse blocks and `indc` determines whether the new matrix will be one- or zero-indexed.
"""
function switch2bsr(csr::CuSparseMatrixCSR, blockDim::Cint, dir::SparseChar='R', inda::SparseChar='O', indc::SparseChar='O') end
