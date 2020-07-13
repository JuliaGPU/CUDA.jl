# wrappers of low-level functionality

import LinearAlgebra: SingularException, HermOrSym, AbstractTriangular, *, +, -, \, mul!

export switch2csr, switch2csc, switch2bsr
export axpyi!, axpyi, sctr!, sctr, gthr!, gthr, gthrz!, grthrz, roti!, roti
export sv2!, sv2, sv_solve!, sv
export mm2!, mm2
export ic02!, ic02, ilu02!, ilu02


# essentials

function cusparseCreate()
    handle = Ref{cusparseHandle_t}()
    res = @retry_reclaim err->isequal(err, CUSPARSE_STATUS_ALLOC_FAILED) ||
                              isequal(err, CUSPARSE_STATUS_NOT_INITIALIZED) begin
        unsafe_cusparseCreate(handle)
    end
    if res != CUSPARSE_STATUS_SUCCESS
         throw_api_error(res)
    end
    handle[]
end

function cusparseGetProperty(property::libraryPropertyType)
    value_ref = Ref{Cint}()
    cusparseGetProperty(property, value_ref)
    value_ref[]
end

version() = VersionNumber(cusparseGetProperty(CUDA.MAJOR_VERSION),
                          cusparseGetProperty(CUDA.MINOR_VERSION),
                          cusparseGetProperty(CUDA.PATCH_LEVEL))


# sparse matrix descriptor

mutable struct CuSparseMatrixDescriptor
    handle::cusparseMatDescr_t

    function CuSparseMatrixDescriptor()
        descr_ref = Ref{cusparseMatDescr_t}()
        cusparseCreateMatDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(cusparseDestroyMatDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseMatDescr_t}, desc::CuSparseMatrixDescriptor) = desc.handle

function CuSparseMatrixDescriptor(MatrixType, FillMode, DiagType, IndexBase)
    desc = CuSparseMatrixDescriptor()
    if MatrixType != CUSPARSE_MATRIX_TYPE_GENERAL
        cusparseSetMatType(desc, MatrixType)
    end
    cusparseSetMatFillMode(desc, FillMode)
    cusparseSetMatDiagType(desc, DiagType)
    if IndexBase != CUSPARSE_INDEX_BASE_ZERO
        cusparseSetMatIndexBase(desc, IndexBase)
    end
    return desc
end


# Type conversion

for (fname,elty) in ((:cusparseScsr2csc, :Float32),
                     (:cusparseDcsr2csc, :Float64),
                     (:cusparseCcsr2csc, :ComplexF32),
                     (:cusparseZcsr2csc, :ComplexF64))
    @eval begin
        function switch2csc(csr::CuSparseMatrixCSR{$elty},inda::SparseChar='O')
            cuind = cusparseindex(inda)
            m,n = csr.dims
            colPtr = CUDA.zeros(Cint, n+1)
            rowVal = CUDA.zeros(Cint, csr.nnz)
            nzVal = CUDA.zeros($elty, csr.nnz)
            if version() >= v"10.2"
                # TODO: algorithm configuratibility?
                @workspace size=@argout(
                        cusparseCsr2cscEx2_bufferSize(handle(), m, n, csr.nnz, csr.nzVal,
                            csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                            cudaDataType($elty), CUSPARSE_ACTION_NUMERIC, cuind,
                            CUSPARSE_CSR2CSC_ALG1, out(Ref{Csize_t}(1)))
                    )[] buffer->begin
                        cusparseCsr2cscEx2(handle(), m, n, csr.nnz, csr.nzVal,
                            csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal, 
                            cudaDataType($elty), CUSPARSE_ACTION_NUMERIC, cuind,
                            CUSPARSE_CSR2CSC_ALG1, buffer)
                    end
            else
                $fname(handle(), m, n, csr.nnz, csr.nzVal,
                    csr.rowPtr, csr.colVal, nzVal, rowVal,
                    colPtr, CUSPARSE_ACTION_NUMERIC, cuind)
            end
            csc = CuSparseMatrixCSC(colPtr,rowVal,nzVal,csr.nnz,csr.dims)
            csc
        end
        function switch2csr(csc::CuSparseMatrixCSC{$elty},inda::SparseChar='O')
            cuind  = cusparseindex(inda)
            m,n    = csc.dims
            rowPtr = CUDA.zeros(Cint,m+1)
            colVal = CUDA.zeros(Cint,csc.nnz)
            nzVal  = CUDA.zeros($elty,csc.nnz)
            if version() >= v"10.2"
                # TODO: algorithm configuratibility?
                @workspace size=@argout(
                        cusparseCsr2cscEx2_bufferSize(handle(), n, m, csc.nnz, csc.nzVal,
                            csc.colPtr, csc.rowVal, nzVal, rowPtr, colVal,
                            cudaDataType($elty), CUSPARSE_ACTION_NUMERIC, cuind,
                            CUSPARSE_CSR2CSC_ALG1, out(Ref{Csize_t}(1)))
                    )[] buffer->begin
                        cusparseCsr2cscEx2(handle(), n, m, csc.nnz, csc.nzVal,
                            csc.colPtr, csc.rowVal, nzVal, rowPtr, colVal,
                            cudaDataType($elty), CUSPARSE_ACTION_NUMERIC, cuind,
                            CUSPARSE_CSR2CSC_ALG1, buffer)
                    end
            else
                $fname(handle(), n, m, csc.nnz, csc.nzVal,
                    csc.colPtr, csc.rowVal, nzVal, colVal,
                    rowPtr, CUSPARSE_ACTION_NUMERIC, cuind)
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
            cudir = cusparsedir(dir)
            cuinda = cusparseindex(inda)
            cuindc = cusparseindex(indc)
            m,n = csr.dims
            nnz = Ref{Cint}(1)
            mb = div((m + blockDim - 1),blockDim)
            nb = div((n + blockDim - 1),blockDim)
            bsrRowPtr = CUDA.zeros(Cint,mb + 1)
            cudesca = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            cusparseXcsr2bsrNnz(handle(), cudir, m, n, cudesca, csr.rowPtr,
                                csr.colVal, blockDim, cudescc, bsrRowPtr, nnz)
            bsrNzVal = CUDA.zeros($elty, nnz[] * blockDim * blockDim )
            bsrColInd = CUDA.zeros(Cint, nnz[])
            $fname(handle(), cudir, m, n,
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
            cudir = cusparsedir(bsr.dir)
            cuinda = cusparseindex(inda)
            cuindc = cusparseindex(indc)
            m,n = bsr.dims
            mb = div(m,bsr.blockDim)
            nb = div(n,bsr.blockDim)
            nnz = bsr.nnz * bsr.blockDim * bsr.blockDim
            cudesca = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            csrRowPtr = CUDA.zeros(Cint, m + 1)
            csrColInd = CUDA.zeros(Cint, nnz)
            csrNzVal  = CUDA.zeros($elty, nnz)
            $fname(handle(), cudir, mb, nb,
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
            cuind = cusparseindex(ind)
            m,n = csr.dims
            denseA = CUDA.zeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            $rname(handle(), m, n, cudesc, csr.nzVal,
                   csr.rowPtr, csr.colVal, denseA, lda)
            denseA
        end
        function Base.Array(csc::CuSparseMatrixCSC{$elty},ind::SparseChar='O')
            cuind = cusparseindex(ind)
            m,n = csc.dims
            denseA = CUDA.zeros($elty,m,n)
            lda = max(1,stride(denseA,2))
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
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
            cuind = cusparseindex(ind)
            cudir = cusparsedir('R')
            if fmt == 'C'
                cudir = cusparsedir(fmt)
            end
            m,n = size(A)
            lda = max(1,stride(A,2))
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            nnzRowCol = CUDA.zeros(Cint, fmt == 'R' ? m : n)
            nnzTotal = Ref{Cint}(1)
            $nname(handle(),
                   cudir, m, n, cudesc, A, lda, nnzRowCol,
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



# Level 1 CUSPARSE functions

"""
    axpyi!(alpha::BlasFloat, X::CuSparseVector, Y::CuVector, index::SparseChar)

Computes `alpha * X + Y` for sparse `X` and dense `Y`.
"""
axpyi!(alpha::BlasFloat, X::CuSparseVector, Y::CuVector, index::SparseChar)

for (fname,elty) in ((:cusparseSaxpyi, :Float32),
                     (:cusparseDaxpyi, :Float64),
                     (:cusparseCaxpyi, :ComplexF32),
                     (:cusparseZaxpyi, :ComplexF64))
    @eval begin
        function axpyi!(alpha::$elty,
                        X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            $fname(handle(), X.nnz, [alpha], X.nzVal, X.iPtr, Y, cuind)
            Y
        end
        function axpyi(alpha::$elty,
                       X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            axpyi!(alpha,X,copy(Y),index)
        end
        function axpyi(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            axpyi!(one($elty),X,copy(Y),index)
        end
    end
end

"""
    gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices.
"""
function gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar) end
for (fname,elty) in ((:cusparseSgthr, :Float32),
                     (:cusparseDgthr, :Float64),
                     (:cusparseCgthr, :ComplexF32),
                     (:cusparseZgthr, :ComplexF64))
    @eval begin
        function gthr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            $fname(handle(), X.nnz, Y, X.nzVal, X.iPtr, cuind)
            X
        end
        function gthr(X::CuSparseVector{$elty},
                      Y::CuVector{$elty},
                      index::SparseChar)
            gthr!(copy(X),Y,index)
        end
    end
end

"""
    gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices, and zeros out those elements of `Y`.
"""
function gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar) end
for (fname,elty) in ((:cusparseSgthrz, :Float32),
                     (:cusparseDgthrz, :Float64),
                     (:cusparseCgthrz, :ComplexF32),
                     (:cusparseZgthrz, :ComplexF64))
    @eval begin
        function gthrz!(X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            $fname(handle(), X.nnz, Y, X.nzVal, X.iPtr, cuind)
            X,Y
        end
        function gthrz(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            gthrz!(copy(X),copy(Y),index)
        end
    end
end

"""
    roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar)

Performs the Givens rotation specified by `c` and `s` to sparse `X` and dense `Y`.
"""
function roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar) end
for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       c::$elty,
                       s::$elty,
                       index::SparseChar)
            cuind = cusparseindex(index)
            $fname(handle(), X.nnz, X.nzVal, X.iPtr, Y, [c], [s], cuind)
            X,Y
        end
        function roti(X::CuSparseVector{$elty},
                      Y::CuVector{$elty},
                      c::$elty,
                      s::$elty,
                      index::SparseChar)
            roti!(copy(X),copy(Y),c,s,index)
        end
    end
end

"""
    sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Set `Y[:] = X[:]` for dense `Y` and sparse `X`.
"""
function sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar) end

for (fname,elty) in ((:cusparseSsctr, :Float32),
                     (:cusparseDsctr, :Float64),
                     (:cusparseCsctr, :ComplexF32),
                     (:cusparseZsctr, :ComplexF64))
    @eval begin
        function sctr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            $fname(handle(), X.nnz, X.nzVal, X.iPtr, Y, cuind)
            Y
        end
        function sctr(X::CuSparseVector{$elty},
                      index::SparseChar)
            sctr!(X, CUDA.zeros($elty, X.dims[1]),index)
        end
    end
end

## level 2 functions

"""
    mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, beta::BlasFloat, Y::CuVector, index::SparseChar)

Performs `Y = alpha * op(A) *X + beta * Y`, where `op` can be nothing (`transa = N`), tranpose (`transa = T`)
or conjugate transpose (`transa = C`). `X` is a sparse vector, and `Y` is dense.
"""
function mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector,
             beta::BlasFloat, Y::CuVector, index::SparseChar) end
for (fname,elty) in ((:cusparseSbsrmv, :Float32),
                     (:cusparseDbsrmv, :Float64),
                     (:cusparseCbsrmv, :ComplexF32),
                     (:cusparseZbsrmv, :ComplexF64))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::CuSparseMatrixBSR{$elty},
                     X::CuVector{$elty},
                     beta::$elty,
                     Y::CuVector{$elty},
                     index::SparseChar)
            cudir = cusparsedir(A.dir)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            mb = div(m,A.blockDim)
            nb = div(n,A.blockDim)
            if transa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X,m,Y,n)
            end
            $fname(handle(), cudir, cutransa, mb, nb,
                   A.nnz, [alpha], cudesc, A.nzVal, A.rowPtr,
                   A.colVal, A.blockDim, X, [beta], Y)
            Y
        end
    end
end

"""
    sv2!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar)

Performs `X = alpha * op(A) \\ X `, where `op` can be nothing (`transa = N`), tranpose (`transa = T`)
or conjugate transpose (`transa = C`). `X` is a dense vector, and `uplo` tells `sv2!` which triangle
of the block sparse matrix `A` to reference.
"""
function sv2!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar) end
# bsrsv2
for (bname,aname,sname,elty) in ((:cusparseSbsrsv2_bufferSize, :cusparseSbsrsv2_analysis, :cusparseSbsrsv2_solve, :Float32),
                                 (:cusparseDbsrsv2_bufferSize, :cusparseDbsrsv2_analysis, :cusparseDbsrsv2_solve, :Float64),
                                 (:cusparseCbsrsv2_bufferSize, :cusparseCbsrsv2_analysis, :cusparseCbsrsv2_solve, :ComplexF32),
                                 (:cusparseZbsrsv2_bufferSize, :cusparseZbsrsv2_analysis, :cusparseZbsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixBSR{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)
            cutransa = cusparseop(transa)
            cudir    = cusparsedir(A.dir)
            cuind    = cusparseindex(index)
            cuplo    = cusparsefill(uplo)
            cudesc   = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, cuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n      = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("X must have length $m, but has length $mX"))
            end
            info = bsrsv2Info_t[0]
            cusparseCreateBsrsv2Info(info)
            @workspace size=@argout(
                    $bname(handle(), cudir, cutransa, mb, A.nnz,
                           cudesc, A.nzVal, A.rowPtr, A.colVal, A.blockDim,
                           info[1], out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), cudir, cutransa, mb, A.nnz,
                           cudesc, A.nzVal, A.rowPtr, A.colVal, A.blockDim,
                           info[1], CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), cudir, cutransa, mb, A.nnz,
                           [alpha], cudesc, A.nzVal, A.rowPtr, A.colVal,
                           A.blockDim, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrsv2Info(info[1])
            X
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     alpha::$elty,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,alpha,A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,one($elty),A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     alpha::$elty,
                     A::AbstractTriangular,
                     X::CuVector{$elty},
                     index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            sv2!(transa,uplo,alpha,A.data,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     A::AbstractTriangular,
                     X::CuVector{$elty},
                     index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            sv2!(transa,uplo,one($elty),A.data,copy(X),index)
        end
    end
end

# csrsv2
for (bname,aname,sname,elty) in ((:cusparseScsrsv2_bufferSize, :cusparseScsrsv2_analysis, :cusparseScsrsv2_solve, :Float32),
                                 (:cusparseDcsrsv2_bufferSize, :cusparseDcsrsv2_analysis, :cusparseDcsrsv2_solve, :Float64),
                                 (:cusparseCcsrsv2_bufferSize, :cusparseCcsrsv2_analysis, :cusparseCcsrsv2_solve, :ComplexF32),
                                 (:cusparseZcsrsv2_bufferSize, :cusparseZcsrsv2_analysis, :cusparseZcsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixCSR{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)
            cutransa  = cusparseop(transa)
            cuind     = cusparseindex(index)
            cuuplo    = cusparsefill(uplo)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            @workspace size=@argout(
                    $bname(handle(), cutransa, m, A.nnz,
                           cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), cutransa, m, A.nnz,
                           cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), cutransa, m,
                           A.nnz, [alpha], cudesc, A.nzVal, A.rowPtr,
                           A.colVal, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end

# cscsv2
for (bname,aname,sname,elty) in ((:cusparseScsrsv2_bufferSize, :cusparseScsrsv2_analysis, :cusparseScsrsv2_solve, :Float32),
                                 (:cusparseDcsrsv2_bufferSize, :cusparseDcsrsv2_analysis, :cusparseDcsrsv2_solve, :Float64),
                                 (:cusparseCcsrsv2_bufferSize, :cusparseCcsrsv2_analysis, :cusparseCcsrsv2_solve, :ComplexF32),
                                 (:cusparseZcsrsv2_bufferSize, :cusparseZcsrsv2_analysis, :cusparseZcsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixCSC{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)
            ctransa = 'N'
            cuplo = 'U'
            if transa == 'N'
                ctransa = 'T'
            end
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            @workspace size=@argout(
                    $bname(handle(), cutransa, m, A.nnz,
                           cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), cutransa, m, A.nnz,
                           cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), cutransa, m,
                           A.nnz, [alpha], cudesc, A.nzVal, A.colPtr,
                           A.rowVal, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end


## level 3 functions

"""
    mm2!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)

Multiply the sparse matrix `A` by the dense matrix `B`, filling in dense matrix `C`.
`C = alpha*op(A)*op(B) + beta*C`. `op(A)` can be nothing (`transa = N`), transpose
(`transa = T`), or conjugate transpose (`transa = C`), and similarly for `op(B)` and
`transb`.
"""
function mm2!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar) end
for (fname,elty) in ((:cusparseSbsrmm, :Float32),
                     (:cusparseDbsrmm, :Float64),
                     (:cusparseCbsrmm, :ComplexF32),
                     (:cusparseZbsrmm, :ComplexF64))
    @eval begin
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixBSR{$elty},
                      B::CuMatrix{$elty},
                      beta::$elty,
                      C::CuMatrix{$elty},
                      index::SparseChar)
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,k = A.dims
            mb = div(m,A.blockDim)
            kb = div(k,A.blockDim)
            n = size(C)[2]
            if transa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif transa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif transa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif transa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), cudir,
                   cutransa, cutransb, mb, n, kb, A.nnz,
                   [alpha], cudesc, A.nzVal,A.rowPtr, A.colVal,
                   A.blockDim, B, ldb, [beta], C, ldc)
            C
        end
    end
end

for (fname,elty) in ((:cusparseScsrmm2, :Float32),
                     (:cusparseDcsrmm2, :Float64),
                     (:cusparseCcsrmm2, :ComplexF32),
                     (:cusparseZcsrmm2, :ComplexF64))
    @eval begin
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixCSR{$elty},
                      B::CuMatrix{$elty},
                      beta::$elty,
                      C::CuMatrix{$elty},
                      index::SparseChar)
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,k = A.dims
            n = size(C)[2]
            if transa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif transa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif transa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif transa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(),
                   cutransa, cutransb, m, n, k, A.nnz, [alpha], cudesc,
                   A.nzVal, A.rowPtr, A.colVal, B, ldb, [beta], C, ldc)
            C
        end
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixCSC{$elty},
                      B::CuMatrix{$elty},
                      beta::$elty,
                      C::CuMatrix{$elty},
                      index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            cutransb = cusparseop(transb)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            k,m = A.dims
            n = size(C)[2]
            if ctransa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif ctransa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif ctransa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif ctransa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(),
                   cutransa, cutransb, m, n, k, A.nnz, [alpha], cudesc,
                   A.nzVal, A.colPtr, A.rowVal, B, ldb, [beta], C, ldc)
            C
        end
    end
end

for elty in (:Float32,:Float64,:ComplexF32,:ComplexF64)
    @eval begin
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     alpha::$elty,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     beta::$elty,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            mm2!(transa,transb,alpha,A,B,beta,copy(C),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     beta::$elty,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            mm2(transa,transb,one($elty),A,B,beta,C,index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     C::CuMatrix{$elty},
                     index::SparseChar)
            mm2(transa,transb,one($elty),A,B,one($elty),C,index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     alpha::$elty,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,alpha,A,B,zero($elty),CUDA.zeros($elty,(m,n)),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,one($elty),A,B,zero($elty),CUDA.zeros($elty,(m,n)),index)
        end
    end
end

# bsrsm2
for (bname,aname,sname,elty) in ((:cusparseSbsrsm2_bufferSize, :cusparseSbsrsm2_analysis, :cusparseSbsrsm2_solve, :Float32),
                                 (:cusparseDbsrsm2_bufferSize, :cusparseDbsrsm2_analysis, :cusparseDbsrsm2_solve, :Float64),
                                 (:cusparseCbsrsm2_bufferSize, :cusparseCbsrsm2_analysis, :cusparseCbsrsm2_solve, :ComplexF32),
                                 (:cusparseZbsrsm2_bufferSize, :cusparseZbsrsm2_analysis, :cusparseZbsrsm2_solve, :ComplexF64))
    @eval begin
        function bsrsm2!(transa::SparseChar,
                         transxy::SparseChar,
                         alpha::$elty,
                         A::CuSparseMatrixBSR{$elty},
                         X::CuMatrix{$elty},
                         index::SparseChar)
            cutransa  = cusparseop(transa)
            cutransxy = cusparseop(transxy)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            mb = div(m,A.blockDim)
            mX,nX = size(X)
            if transxy == 'N' && (mX != m)
                throw(DimensionMismatch(""))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = bsrsm2Info_t[0]
            cusparseCreateBsrsm2Info(info)
            @workspace size=@argout(
                    $bname(handle(), cudir, cutransa, cutransxy,
                           mb, nX, A.nnz, cudesc, A.nzVal, A.rowPtr,
                           A.colVal, A.blockDim, info[],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), cudir, cutransa, cutransxy,
                           mb, nX, A.nnz, cudesc, A.nzVal, A.rowPtr,
                           A.colVal, A.blockDim, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrsm2_zeroPivot(handle(), info[], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), cudir, cutransa, cutransxy, mb,
                           nX, A.nnz, [alpha], cudesc, A.nzVal, A.rowPtr,
                           A.colVal, A.blockDim, info[], X, ldx, X, ldx,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrsm2Info(info[1])
            X
        end
        function bsrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::$elty,
                        A::CuSparseMatrixBSR{$elty},
                        X::CuMatrix{$elty},
                        index::SparseChar)
            bsrsm2!(transa,transxy,alpha,A,copy(X),index)
        end
    end
end

# csrsm2
for (bname,aname,sname,elty) in ((:cusparseScsrsm2_bufferSizeExt, :cusparseScsrsm2_analysis, :cusparseScsrsm2_solve, :Float32),
                                 (:cusparseDcsrsm2_bufferSizeExt, :cusparseDcsrsm2_analysis, :cusparseDcsrsm2_solve, :Float64),
                                 (:cusparseCcsrsm2_bufferSizeExt, :cusparseCcsrsm2_analysis, :cusparseCcsrsm2_solve, :ComplexF32),
                                 (:cusparseZcsrsm2_bufferSizeExt, :cusparseZcsrsm2_analysis, :cusparseZcsrsm2_solve, :ComplexF64))
    @eval begin
        function csrsm2!(transa::SparseChar,
                         transxy::SparseChar,
                         alpha::$elty,
                         A::CuSparseMatrixCSR{$elty},
                         X::CuMatrix{$elty},
                         index::SparseChar)
            cutransa  = cusparseop(transa)
            cutransxy = cusparseop(transxy)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            mX,nX = size(X)
            if transxy == 'N' && (mX != m)
                throw(DimensionMismatch(""))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = csrsm2Info_t[0]
            cusparseCreateCsrsm2Info(info)
            # use non block algo (0) for now...
            @workspace size=@argout(
                    $bname(handle(), 0, cutransa, cutransxy,
                           m, nX, A.nnz, [alpha], cudesc, A.nzVal, A.rowPtr,
                           A.colVal, X, ldx, info[], CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                           out(Ref{UInt64}(1)))
                )[] buffer->begin
                    $aname(handle(), 0, cutransa, cutransxy,
                           m, nX, A.nnz, [alpha], cudesc, A.nzVal, A.rowPtr,
                           A.colVal, X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsm2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), 0, cutransa, cutransxy, m,
                           nX, A.nnz, [alpha], cudesc, A.nzVal, A.rowPtr,
                           A.colVal, X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsm2Info(info[])
            X
        end
        function csrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::$elty,
                        A::CuSparseMatrixCSR{$elty},
                        X::CuMatrix{$elty},
                        index::SparseChar)
            csrsm2!(transa,transxy,alpha,A,copy(X),index)
        end
    end
end




## preconditioners

"""
    ic02!(A::CuSparseMatrix, index::SparseChar)

Incomplete Cholesky factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
function ic02!(A::CuSparseMatrix, index::SparseChar) end
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSR{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsric02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), m, A.nnz,
                           cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

# cscic02
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSC{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsric02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), m, A.nnz,
                           cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

"""
    ilu02!(A::CuSparseMatrix, index::SparseChar)

Incomplete LU factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
function ilu02!(A::CuSparseMatrix, index::SparseChar) end
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixCSR{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrilu02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), m, A.nnz,
                           cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrilu02Info(info[1])
            A
        end
    end
end

# cscilu02
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixCSC{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, cudesc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrilu02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), m, A.nnz,
                           cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrilu02Info(info[1])
            A
        end
    end
end

# bsric02
for (bname,aname,sname,elty) in ((:cusparseSbsric02_bufferSize, :cusparseSbsric02_analysis, :cusparseSbsric02, :Float32),
                                 (:cusparseDbsric02_bufferSize, :cusparseDbsric02_analysis, :cusparseDbsric02, :Float64),
                                 (:cusparseCbsric02_bufferSize, :cusparseCbsric02_analysis, :cusparseCbsric02, :ComplexF32),
                                 (:cusparseZbsric02_bufferSize, :cusparseZbsric02_analysis, :cusparseZbsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixBSR{$elty},
                       index::SparseChar)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsric02Info_t[0]
            cusparseCreateBsric02Info(info)
            @workspace size=@argout(
                    $bname(handle(), cudir, mb, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), cudir, mb, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsric02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), cudir, mb, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsric02Info(info[1])
            A
        end
    end
end

# bsrilu02
for (bname,aname,sname,elty) in ((:cusparseSbsrilu02_bufferSize, :cusparseSbsrilu02_analysis, :cusparseSbsrilu02, :Float32),
                                 (:cusparseDbsrilu02_bufferSize, :cusparseDbsrilu02_analysis, :cusparseDbsrilu02, :Float64),
                                 (:cusparseCbsrilu02_bufferSize, :cusparseCbsrilu02_analysis, :cusparseCbsrilu02, :ComplexF32),
                                 (:cusparseZbsrilu02_bufferSize, :cusparseZbsrilu02_analysis, :cusparseZbsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixBSR{$elty},
                        index::SparseChar)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = CuSparseMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsrilu02Info_t[0]
            cusparseCreateBsrilu02Info(info)
            @workspace size=@argout(
                    $bname(handle(), cudir, mb, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), cudir, mb, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrilu02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
                    end
                    $sname(handle(), cudir, mb, A.nnz, cudesc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrilu02Info(info[1])
            A
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function ilu02(A::CuSparseMatrix{$elty},
                       index::SparseChar)
            ilu02!(copy(A),index)
        end
        function ic02(A::CuSparseMatrix{$elty},
                      index::SparseChar)
            ic02!(copy(A),index)
        end
        function ilu02(A::HermOrSym{$elty,CuSparseMatrix{$elty}},
                       index::SparseChar)
            ilu02!(copy(A.data),index)
        end
        function ic02(A::HermOrSym{$elty,CuSparseMatrix{$elty}},
                      index::SparseChar)
            ic02!(copy(A.data),index)
        end
    end
end
