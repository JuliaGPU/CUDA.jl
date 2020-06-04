# wrappers of low-level functionality

function cusolverGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cusolverGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cusolverGetProperty(CUDA.MAJOR_VERSION),
                          cusolverGetProperty(CUDA.MINOR_VERSION),
                          cusolverGetProperty(CUDA.PATCH_LEVEL))


#
# Sparse
#

using SparseArrays

using ..CUSPARSE: CuSparseMatrixCSR, CuSparseMatrixCSC, cusparseindex,
                  CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER,
                  CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT,
                  CUSPARSE_DIAG_TYPE_UNIT, cusparseMatDescr

function cusolverSpCreate()
  handle = Ref{cusolverSpHandle_t}()
  cusolverSpCreate(handle)
  return handle[]
end

#csrlsvlu
for (fname, elty, relty) in ((:cusolverSpScsrlsvluHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsvluHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsvluHost, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvluHost, :ComplexF64, Float64))
    @eval begin
        function csrlsvlu!(A::SparseMatrixCSC{$elty},
                           b::Vector{$elty},
                           x::Vector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::Char)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("LU factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            Mat     = similar(A)
            transpose!(Mat, A)
            cudesca = cusparseMatDescr(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref(cudesca)
            singularity = Ref{Cint}(1)
            $fname(sparse_handle(), n, length(A.nzval), rcudesca, Mat.nzval,
                   convert(Vector{Cint},Mat.colptr), convert(Vector{Cint},Mat.rowval), b,
                   tol, reorder, x, singularity)

            if singularity[] != -1
                throw(SingularException(singularity[]))
            end

            x
        end
    end
end

#csrlsvqr
for (fname, elty, relty) in ((:cusolverSpScsrlsvqr, :Float32, :Float32),
                             (:cusolverSpDcsrlsvqr, :Float64, :Float64),
                             (:cusolverSpCcsrlsvqr, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvqr, :ComplexF64, Float64))
    @eval begin
        function csrlsvqr!(A::CuSparseMatrixCSR{$elty},
                           b::CuVector{$elty},
                           x::CuVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::Char)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("QR factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            cudesca = cusparseMatDescr(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref(cudesca)
            singularity = Ref{Cint}(1)
            $fname(sparse_handle(), n, A.nnz, rcudesca, A.nzVal, A.rowPtr, A.colVal, b, tol, reorder, x, singularity)

            if singularity[] != -1
                throw(SingularException(singularity[]))
            end

            x
        end
    end
end

#csrlsvchol
for (fname, elty, relty) in ((:cusolverSpScsrlsvchol, :Float32, :Float32),
                             (:cusolverSpDcsrlsvchol, :Float64, :Float64),
                             (:cusolverSpCcsrlsvchol, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvchol, :ComplexF64, Float64))
    @eval begin
        function csrlsvchol!(A::CuSparseMatrixCSR{$elty},
                             b::CuVector{$elty},
                             x::CuVector{$elty},
                             tol::$relty,
                             reorder::Cint,
                             inda::Char)
            cuinda = cusparseindex(inda)
            n      = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            cudesca = cusparseMatDescr(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref(cudesca)
            singularity = zeros(Cint,1)
            $fname(sparse_handle(), n, A.nnz, rcudesca, A.nzVal, A.rowPtr, A.colVal, b, tol, reorder, x, singularity)

            if singularity[1] != -1
                throw(SingularException(singularity[1]))
            end

            x
        end
    end
end

#csrlsqvqr
for (fname, elty, relty) in ((:cusolverSpScsrlsqvqrHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsqvqrHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsqvqrHost, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsqvqrHost, :ComplexF64, Float64))
    @eval begin
        function csrlsqvqr!(A::SparseMatrixCSC{$elty},
                            b::Vector{$elty},
                            x::Vector{$elty},
                            tol::$relty,
                            inda::Char)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m < n
                throw(DimensionMismatch("csrlsqvqr only works when the first dimension of A, $m, is greater than or equal to the second dimension of A, $n"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            cudesca  = cusparseMatDescr(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref(cudesca)
            p        = zeros(Cint,n)
            min_norm = zeros($relty,1)
            rankA    = zeros(Cint,1)
            Mat      = similar(A)
            transpose!(Mat, A)
            $fname(sparse_handle(), m, n, length(A.nzval), rcudesca, Mat.nzval,
                   convert(Vector{Cint},Mat.colptr), convert(Vector{Cint},Mat.rowval),
                   b, tol, rankA, x, p, min_norm)

            x, rankA[1], p, min_norm[1]
        end
    end
end

#csreigvsi
for (fname, elty, relty) in ((:cusolverSpScsreigvsi, :Float32, :Float32),
                             (:cusolverSpDcsreigvsi, :Float64, :Float64),
                             (:cusolverSpCcsreigvsi, :ComplexF32, :Float32),
                             (:cusolverSpZcsreigvsi, :ComplexF64, Float64))
    @eval begin
        function csreigvsi(A::CuSparseMatrixCSR{$elty},
                           μ_0::$elty,
                           x_0::CuVector{$elty},
                           tol::$relty,
                           maxite::Cint,
                           inda::Char)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            if n != length(x_0)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of x_0, $(length(x_0))"))
            end

            cudesca = cusparseMatDescr(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref(cudesca)
            x       = copy(x_0)
            μ       = CUDA.zeros($elty,1)
            $fname(sparse_handle(), n, A.nnz, rcudesca, A.nzVal, A.rowPtr, A.colVal,
                   μ_0, x_0, maxite, tol, μ, x)

            collect(μ)[1], x
        end
    end
end

#csreigs
for (fname, elty, relty) in ((:cusolverSpScsreigsHost, :ComplexF32, :Float32),
                             (:cusolverSpDcsreigsHost, :ComplexF64, :Float64),
                             (:cusolverSpCcsreigsHost, :ComplexF32, :ComplexF32),
                             (:cusolverSpZcsreigsHost, :ComplexF64, :ComplexF64))
    @eval begin
        function csreigs(A::SparseMatrixCSC{$relty},
                         lbc::$elty,
                         ruc::$elty,
                         inda::Char)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end

            cudesca = cusparseMatDescr(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref(cudesca)
            numeigs = zeros(Cint,1)
            Mat     = similar(A)
            transpose!(Mat, A)
            $fname(sparse_handle(), n, length(A.nzval), rcudesca, Mat.nzval,
                   convert(Vector{Cint},Mat.colptr), convert(Vector{Cint},Mat.rowval),
                   lbc, ruc, numeigs)

            numeigs[1]
        end
    end
end



#
# Dense
#

using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using LinearAlgebra.LAPACK: chkargsok, chklapackerror

using ..CUBLAS: cublasfill, cublasop, cublasside, unsafe_batch

function cusolverDnCreate()
  handle = Ref{cusolverDnHandle_t}()
  cusolverDnCreate(handle)
  return handle[]
end

# po
## potrf
for (bname, fname,elty) in ((:cusolverDnSpotrf_bufferSize, :cusolverDnSpotrf, :Float32),
                            (:cusolverDnDpotrf_bufferSize, :cusolverDnDpotrf, :Float64),
                            (:cusolverDnCpotrf_bufferSize, :cusolverDnCpotrf, :ComplexF32),
                            (:cusolverDnZpotrf_bufferSize, :cusolverDnZpotrf, :ComplexF64))
    @eval begin
        function LinearAlgebra.LAPACK.potrf!(uplo::Char,
                        A::CuMatrix{$elty})
            cuuplo  = cublasfill(uplo)
            n       = checksquare(A)
            lda     = max(1, stride(A, 2))

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), cuuplo, n, A, lda, out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), cuuplo, n, A, lda, buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            chkargsok(BlasInt(info))

            A, info
        end
    end
end

## potrs
for (fname,elty) in ((:cusolverDnSpotrs, :Float32),
                     (:cusolverDnDpotrs, :Float64),
                     (:cusolverDnCpotrs, :ComplexF32),
                     (:cusolverDnZpotrs, :ComplexF64))
    @eval begin
        function LinearAlgebra.LAPACK.potrs!(uplo::Char,
                        A::CuMatrix{$elty},
                        B::CuVecOrMat{$elty})
            cuuplo = cublasfill(uplo)
            n = checksquare(A)
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B,2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))

            devinfo = CuArray{Cint}(undef, 1)
            $fname(dense_handle(), cuuplo, n, nrhs, A, lda, B, ldb, devinfo)

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            chkargsok(BlasInt(info))

            B
        end
    end
end

## potri
for (bname, fname,elty) in ((:cusolverDnSpotri_bufferSize, :cusolverDnSpotri, :Float32),
                     (:cusolverDnDpotri_bufferSize, :cusolverDnDpotri, :Float64),
                     (:cusolverDnCpotri_bufferSize, :cusolverDnCpotri, :ComplexF32),
                     (:cusolverDnZpotri_bufferSize, :cusolverDnZpotri, :ComplexF64))
@eval begin
    function LinearAlgebra.LAPACK.potri!(uplo::Char,
                    A::CuMatrix{$elty})

           cuuplo  = cublasfill(uplo)
           n       = checksquare(A)
           lda     = max(1, stride(A, 2))
           devinfo = CuArray{Cint}(undef, 1)

           @workspace eltyp=$elty size=@argout(
                   $bname(dense_handle(), cuuplo, n, A, lda, out(Ref{Cint}(0)))
               )[] buffer->begin
                   $fname(dense_handle(), cuuplo, n, A, lda, buffer, length(buffer), devinfo)
               end

           info = @allowscalar devinfo[1]
           unsafe_free!(devinfo)
           chkargsok(BlasInt(info))

           A
        end
    end
end
"""
    potri!(uplo::Char, A::CuMatrix)

!!! note

    `potri!` requires CUDA >= 10.1
"""
LinearAlgebra.LAPACK.potri!(uplo::Char, A::CuMatrix)

#getrf
for (bname, fname,elty) in ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
                            (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64),
                            (:cusolverDnCgetrf_bufferSize, :cusolverDnCgetrf, :ComplexF32),
                            (:cusolverDnZgetrf_bufferSize, :cusolverDnZgetrf, :ComplexF64))
    @eval begin
        function getrf!(A::CuMatrix{$elty})
            m,n     = size(A)
            lda     = max(1, stride(A, 2))

            devipiv = CuArray{Cint}(undef, min(m,n))
            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), m, n, A, lda, out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), m, n, A, lda, buffer, devipiv, devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            elseif info > 0
                throw(LinearAlgebra.SingularException(info))
            end

            A, devipiv
        end
    end
end

#geqrf
for (bname, fname,elty) in ((:cusolverDnSgeqrf_bufferSize, :cusolverDnSgeqrf, :Float32),
                            (:cusolverDnDgeqrf_bufferSize, :cusolverDnDgeqrf, :Float64),
                            (:cusolverDnCgeqrf_bufferSize, :cusolverDnCgeqrf, :ComplexF32),
                            (:cusolverDnZgeqrf_bufferSize, :cusolverDnZgeqrf, :ComplexF64))
    @eval begin
        function geqrf!(A::CuMatrix{$elty})
            m, n    = size(A)
            lda     = max(1, stride(A, 2))

            tau  = CuArray{$elty}(undef, min(m, n))
            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                $bname(dense_handle(), m, n, A, lda, out(Ref{Cint}(0))))[] buffer->begin
                $fname(dense_handle(), m, n, A, lda, tau, buffer, length(buffer), devinfo)
            end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            A, tau
        end
    end
end

#sytrf
for (bname, fname,elty) in ((:cusolverDnSsytrf_bufferSize, :cusolverDnSsytrf, :Float32),
                            (:cusolverDnDsytrf_bufferSize, :cusolverDnDsytrf, :Float64),
                            (:cusolverDnCsytrf_bufferSize, :cusolverDnCsytrf, :ComplexF32),
                            (:cusolverDnZsytrf_bufferSize, :cusolverDnZsytrf, :ComplexF64))
    @eval begin
        function sytrf!(uplo::Char,
                        A::CuMatrix{$elty})
            cuuplo = cublasfill(uplo)
            n      = checksquare(A)
            lda = max(1, stride(A, 2))

            devipiv = CuArray{Cint}(undef, n)
            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), n, A, lda, out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), cuuplo, n, A, lda, devipiv, buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            elseif info > 0
                throw(LinearAlgebra.SingularException(info))
            end

            A, devipiv
        end
    end
end

#getrs
for (fname,elty) in ((:cusolverDnSgetrs, :Float32),
                     (:cusolverDnDgetrs, :Float64),
                     (:cusolverDnCgetrs, :ComplexF32),
                     (:cusolverDnZgetrs, :ComplexF64))
    @eval begin
        function getrs!(trans::Char,
                        A::CuMatrix{$elty},
                        ipiv::CuVector{Cint},
                        B::CuVecOrMat{$elty})
            cutrans = cublasop(trans)
            n = size(A, 1)
            if size(A, 2) != n
                throw(DimensionMismatch("LU factored matrix A must be square!"))
            end
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B, 2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))

            devinfo = CuArray{Cint}(undef, 1)
            $fname(dense_handle(), cutrans, n, nrhs, A, lda, ipiv, B, ldb, devinfo)

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end
            B
        end
    end
end

#ormqr
for (bname, fname, elty) in ((:cusolverDnSormqr_bufferSize, :cusolverDnSormqr, :Float32),
                             (:cusolverDnDormqr_bufferSize, :cusolverDnDormqr, :Float64),
                             (:cusolverDnCunmqr_bufferSize, :cusolverDnCunmqr, :ComplexF32),
                             (:cusolverDnZunmqr_bufferSize, :cusolverDnZunmqr, :ComplexF64))
    @eval begin
        function ormqr!(side::Char,
                        trans::Char,
                        A::CuMatrix{$elty},
                        tau::CuVector{$elty},
                        C::CuVecOrMat{$elty})
            cutrans = cublasop(trans)
            cuside  = cublasside(side)
            if side == 'L'
                m   = size(A, 1)
                ldc = size(C, 1)
                n   = size(C, 2)
                if m > ldc
                    Ctemp = CUDA.zeros($elty, m - ldc, n)
                    C = [C; Ctemp]
                    ldc = m
                end
                lda = m
            else
                m   = size(C, 1)
                n   = size(C, 2)
                ldc = m
                lda = n
            end
            k       = length(tau)

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), cuside, cutrans, m, n, k, A, lda, tau, C, ldc,
                        out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), cuside, cutrans, m, n, k, A, lda, tau, C, ldc,
                        buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            side == 'L' ? C : C[:, 1:minimum(size(A))]
        end
    end
end

#orgqr
for (bname, fname, elty) in ((:cusolverDnSorgqr_bufferSize, :cusolverDnSorgqr, :Float32),
                             (:cusolverDnDorgqr_bufferSize, :cusolverDnDorgqr, :Float64),
                             (:cusolverDnCungqr_bufferSize, :cusolverDnCungqr, :ComplexF32),
                             (:cusolverDnZungqr_bufferSize, :cusolverDnZungqr, :ComplexF64))
    @eval begin
        function orgqr!(A::CuMatrix{$elty}, tau::CuVector{$elty})
            m = size(A , 1)
            n = min(m, size(A, 2))
            lda = max(1, stride(A, 2))
            k = length(tau)

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), m, n, k, A, lda, tau, out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), m, n, k, A, lda, tau, buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            if n < size(A, 2)
                A[:, 1:n]
            else
                A
            end
        end
    end
end

#gebrd
for (bname, fname, elty, relty) in ((:cusolverDnSgebrd_bufferSize, :cusolverDnSgebrd, :Float32, :Float32),
                                    (:cusolverDnDgebrd_bufferSize, :cusolverDnDgebrd, :Float64, :Float64),
                                    (:cusolverDnCgebrd_bufferSize, :cusolverDnCgebrd, :ComplexF32, :Float32),
                                    (:cusolverDnZgebrd_bufferSize, :cusolverDnZgebrd, :ComplexF64, :Float64))
    @eval begin
        function gebrd!(A::CuMatrix{$elty})
            m, n    = size(A)
            lda     = max(1, stride(A, 2))

            devinfo = CuArray{Cint}(undef, 1)
            k       = min(m, n)
            D       = CuArray{$relty}(undef, k)
            E       = CUDA.zeros($relty, k)
            TAUQ    = CuArray{$elty}(undef, k)
            TAUP    = CuArray{$elty}(undef, k)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), m, n, out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), m, n, A, lda, D, E, TAUQ, TAUP, buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            A, D, E, TAUQ, TAUP
        end
    end
end

for (bname, fname, elty, relty) in ((:cusolverDnSgesvd_bufferSize, :cusolverDnSgesvd, :Float32, :Float32),
                                    (:cusolverDnDgesvd_bufferSize, :cusolverDnDgesvd, :Float64, :Float64),
                                    (:cusolverDnCgesvd_bufferSize, :cusolverDnCgesvd, :ComplexF32, :Float32),
                                    (:cusolverDnZgesvd_bufferSize, :cusolverDnZgesvd, :ComplexF64, :Float64))
    @eval begin
        function gesvd!(jobu::Char,
                        jobvt::Char,
                        A::CuMatrix{$elty})
            m, n    = size(A)
            if m < n
                throw(ArgumentError("CUSOLVER's gesvd currently requires m >= n"))
            end
            lda     = max(1, stride(A, 2))

            if jobu === 'S' && m > n
                U   = CuArray{$elty}(undef, m, n)
            elseif jobu === 'N'
                U   = CuArray{$elty}(undef, m, 0)
            else
                U   = CuArray{$elty}(undef, m, m)
            end
            ldu     = max(1, stride(U, 2))

            S       = CuArray{$relty}(undef, min(m, n))

            if jobvt === 'S' && m < n
                Vt  = CuArray{$elty}(undef, m, n)
            elseif jobvt === 'N'
                Vt  = CuArray{$elty}(undef, 0, n)
            else
                Vt  = CuArray{$elty}(undef, n, n)
            end
            ldvt    = max(1, stride(Vt, 2))

            rwork   = CuArray{$relty}(undef, min(m, n) - 1)
            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), m, n, out(Ref{Cint}(0)))
                )[] work->begin
                    $fname(dense_handle(), jobu, jobvt, m, n, A, lda, S, U, ldu, Vt, ldvt,
                        work, sizeof(work), rwork, devinfo)
                end
            unsafe_free!(rwork)

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            return U, S, Vt
        end
    end
end

for (bname, fname, elty, relty) in ((:cusolverDnSgesvdj_bufferSize, :cusolverDnSgesvdj, :Float32, :Float32),
                                    (:cusolverDnDgesvdj_bufferSize, :cusolverDnDgesvdj, :Float64, :Float64),
                                    (:cusolverDnCgesvdj_bufferSize, :cusolverDnCgesvdj, :ComplexF32, :Float32),
                                    (:cusolverDnZgesvdj_bufferSize, :cusolverDnZgesvdj, :ComplexF64, :Float64))
    @eval begin
        function gesvdj!(jobz::Char,
                         econ::Int,
                         A::CuMatrix{$elty};
                         tol::$relty=eps($relty),
                         max_sweeps::Int=100)
            cujobz  = cusolverjob(jobz)
            m,n     = size(A)
            lda     = max(1, stride(A, 2))

            # Warning! For some reason, the solver needs to access U and V even
            # when only the values are requested
            if jobz === 'V' && econ == 1 && m > n
                U   = CuArray{$elty}(undef, m, n)
            else
                U   = CuArray{$elty}(undef, m, m)
            end
            ldu     = max(1, stride(U, 2))

            S       = CuArray{$relty}(undef, min(m, n))

            if jobz === 'V' && econ == 1 && m < n
                V   = CuArray{$elty}(undef, n, m)
            else
                V   = CuArray{$elty}(undef, n, n)
            end
            ldv     = max(1, stride(V, 2))

            params  = Ref{gesvdjInfo_t}(C_NULL)
            cusolverDnCreateGesvdjInfo(params)
            cusolverDnXgesvdjSetTolerance(params[], tol)
            cusolverDnXgesvdjSetMaxSweeps(params[], max_sweeps)

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), cujobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                        out(Ref{Cint}(0)), params[])
                )[] work->begin
                    $fname(dense_handle(), cujobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                        work, sizeof(work), devinfo, params[])
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            cusolverDnDestroyGesvdjInfo(params[])

            U, S, V
        end
    end
end

for (jname, bname, fname, elty, relty) in ((:syevd!, :cusolverDnSsyevd_bufferSize, :cusolverDnSsyevd, :Float32, :Float32),
                                           (:syevd!, :cusolverDnDsyevd_bufferSize, :cusolverDnDsyevd, :Float64, :Float64),
                                           (:heevd!, :cusolverDnCheevd_bufferSize, :cusolverDnCheevd, :ComplexF32, :Float32),
                                           (:heevd!, :cusolverDnZheevd_bufferSize, :cusolverDnZheevd, :ComplexF64, :Float64))
    @eval begin
        function $jname(jobz::Char,
                        uplo::Char,
                        A::CuMatrix{$elty})
            cuuplo  = cublasfill(uplo)
            cujobz  = cusolverjob(jobz)
            n       = checksquare(A)
            lda     = max(1, stride(A, 2))
            W       = CuArray{$relty}(undef, n)

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), cujobz, cuuplo, n, A, lda, W,
                        out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), cujobz, cuuplo, n, A, lda, W,
                        buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            if jobz == 'N'
                return W
            elseif jobz == 'V'
                return W, A
            end
        end
    end
end

for (jname, bname, fname, elty, relty) in ((:sygvd!, :cusolverDnSsygvd_bufferSize, :cusolverDnSsygvd, :Float32, :Float32),
                                           (:sygvd!, :cusolverDnDsygvd_bufferSize, :cusolverDnDsygvd, :Float64, :Float64),
                                           (:hegvd!, :cusolverDnChegvd_bufferSize, :cusolverDnChegvd, :ComplexF32, :Float32),
                                           (:hegvd!, :cusolverDnZhegvd_bufferSize, :cusolverDnZhegvd, :ComplexF64, :Float64))
    @eval begin
        function $jname(itype::Int,
                        jobz::Char,
                        uplo::Char,
                        A::CuMatrix{$elty},
                        B::CuMatrix{$elty})
            cuuplo  = cublasfill(uplo)
            cujobz  = cusolverjob(jobz)
            nA, nB  = checksquare(A, B)
            if nB != nA
                throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
            end
            n       = nA
            lda     = max(1, stride(A, 2))
            ldb     = max(1, stride(B, 2))
            W       = CuArray{$relty}(undef, n)

            cuitype = cusolverEigType_t(itype)

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), cuitype, cujobz, cuuplo, n, A, lda, B, ldb, W,
                        out(Ref{Cint}(0)))
                )[] buffer->begin
                    $fname(dense_handle(), cuitype, cujobz, cuuplo, n, A, lda, B, ldb, W,
                        buffer, length(buffer), devinfo)
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            if jobz == 'N'
                return W
            elseif jobz == 'V'
                return W, A, B
            end
        end
    end
end

for (jname, bname, fname, elty, relty) in ((:sygvj!, :cusolverDnSsygvj_bufferSize, :cusolverDnSsygvj, :Float32, :Float32),
                                           (:sygvj!, :cusolverDnDsygvj_bufferSize, :cusolverDnDsygvj, :Float64, :Float64),
                                           (:hegvj!, :cusolverDnChegvj_bufferSize, :cusolverDnChegvj, :ComplexF32, :Float32),
                                           (:hegvj!, :cusolverDnZhegvj_bufferSize, :cusolverDnZhegvj, :ComplexF64, :Float64))
    @eval begin
        function $jname(itype::Int,
                        jobz::Char,
                        uplo::Char,
                        A::CuMatrix{$elty},
                        B::CuMatrix{$elty};
                        tol::$relty=eps($relty),
                        max_sweeps::Int=100)
            cujobz  = cusolverjob(jobz)
            cuuplo  = cublasfill(uplo)
            nA, nB  = checksquare(A, B)
            if nB != nA
                throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
            end
            n       = nA
            lda     = max(1, stride(A, 2))
            ldb     = max(1, stride(B, 2))
            W       = CuArray{$relty}(undef, n)
            params  = Ref{syevjInfo_t}(C_NULL)
            cusolverDnCreateSyevjInfo(params)
            cusolverDnXsyevjSetTolerance(params[], tol)
            cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)

            cuitype = cusolverEigType_t(itype)

            devinfo = CuArray{Cint}(undef, 1)
            @workspace eltyp=$elty size=@argout(
                    $bname(dense_handle(), cuitype, cujobz, cuuplo, n, A, lda, B, ldb, W,
                        out(Ref{Cint}(0)), params[])
                )[] buffer->begin
                    $fname(dense_handle(), cuitype, cujobz, cuuplo, n, A, lda, B, ldb, W,
                        buffer, length(buffer), devinfo, params[])
                end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            cusolverDnDestroySyevjInfo(params[])

            if jobz == 'N'
                return W
            elseif jobz == 'V'
                return W, A, B
            end
        end
    end
end

for (jname, bname, fname, elty, relty) in ((:syevjBatched!, :cusolverDnSsyevjBatched_bufferSize, :cusolverDnSsyevjBatched, :Float32, :Float32),
                                           (:syevjBatched!, :cusolverDnDsyevjBatched_bufferSize, :cusolverDnDsyevjBatched, :Float64, :Float64),
                                           (:heevjBatched!, :cusolverDnCheevjBatched_bufferSize, :cusolverDnCheevjBatched, :ComplexF32, :Float32),
                                           (:heevjBatched!, :cusolverDnZheevjBatched_bufferSize, :cusolverDnZheevjBatched, :ComplexF64, :Float64)
                                           )
    @eval begin
        function $jname(jobz::Char,
                        uplo::Char,
                        A::CuArray{$elty};
                        tol::$relty=eps($relty),
                        max_sweeps::Int=100)

            # Set up information for the solver arguments
            cuuplo  = cublasfill(uplo)
            cujobz  = cusolverjob(jobz)
            n       = checksquare(A)
            lda     = max(1, stride(A, 2))
            batchSize = size(A,3)
            W       = CuArray{$relty}(undef, n,batchSize)
            params  = Ref{syevjInfo_t}(C_NULL)
            devinfo = CuArray{Cint}(undef, batchSize)

            # Initialize the solver parameters
            cusolverDnCreateSyevjInfo(params)
            cusolverDnXsyevjSetTolerance(params[], tol)
            cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)

            # Calculate the workspace size
            lwork = @argout(CUSOLVER.$bname(dense_handle(), cujobz, cuuplo, n,
                            A, lda, W, out(Ref{Cint}(0)), params, batchSize))[]

            # Run the solver
            @workspace eltyp=$elty size=lwork work->begin
                $fname(dense_handle(), cujobz, cuuplo, n, A, lda, W, work,
                       lwork, devinfo, params[], batchSize)
            end

            # Copy the solver info and delete the device memory
            info = @allowscalar collect(devinfo)
            unsafe_free!(devinfo)

            # Double check the solver's exit status
            for i = 1:batchSize
                if info[i] < 0
                    throw(ArgumentError("The $(info)th parameter of the $(i)th solver is wrong"))
                end
            end

            # Return eigenvalues (in W) and possibly eigenvectors (in A)
            if jobz == 'N'
                return W
            elseif jobz == 'V'
                return W, A
            end
        end
    end
end

for (jname, fname, elty) in ((:potrsBatched!, :cusolverDnSpotrsBatched, :Float32),
                             (:potrsBatched!, :cusolverDnDpotrsBatched, :Float64),
                             (:potrsBatched!, :cusolverDnCpotrsBatched, :ComplexF32),
                             (:potrsBatched!, :cusolverDnZpotrsBatched, :ComplexF64)
                             )
    @eval begin
        # cusolverStatus_t
        # cusolverDnSpotrsBatched(
        #     cusolverDnHandle_t handle,
        #     cublasFillMode_t uplo,
        #     int n,
        #     int nrhs,
        #     float *Aarray[],
        #     int lda,
        #     float *Barray[],
        #     int ldb,
        #     int *info,
        #     int batchSize);
        function $jname(uplo::Char, A::Vector{<:CuMatrix{$elty}}, B::Vector{<:CuVecOrMat{$elty}})
            if length(A) != length(B)
                throw(DimensionMismatch(""))
            end
            # Set up information for the solver arguments
            cuuplo = cublasfill(uplo)
            n = checksquare(A[1])
            if size(B[1], 1) != n
                throw(DimensionMismatch("first dimension of B[i], $(size(B[1],1)), must match second dimension of A, $n"))
            end
            nrhs = size(B[1], 2)
            # cuSOLVER's Remark 1: only nrhs=1 is supported.
            if nrhs != 1
                throw(ArgumentError("cuSOLVER only supports vectors for B"))
            end
            lda = max(1, stride(A[1], 2))
            ldb = max(1, stride(B[1], 2))
            batchSize = length(A)
            devinfo = CuArray{Cint}(undef, 1)

            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)

            # Run the solver
            $fname(dense_handle(), cuuplo, n, nrhs, Aptrs, lda, Bptrs, ldb, devinfo, batchSize)

            # Copy the solver info and delete the device memory
            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            chklapackerror(BlasInt(info))

            return B
        end
    end
end

for (jname, fname, elty) in ((:potrfBatched!, :cusolverDnSpotrfBatched, :Float32),
                             (:potrfBatched!, :cusolverDnDpotrfBatched, :Float64),
                             (:potrfBatched!, :cusolverDnCpotrfBatched, :ComplexF32),
                             (:potrfBatched!, :cusolverDnZpotrfBatched, :ComplexF64)
                             )
    @eval begin
        # cusolverStatus_t
        # cusolverDnSpotrfBatched(
        #     cusolverDnHandle_t handle,
        #     cublasFillMode_t uplo,
        #     int n,
        #     float *Aarray[],
        #     int lda,
        #     int *infoArray,
        #     int batchSize);
        function $jname(uplo::Char, A::Vector{<:CuMatrix{$elty}})

            # Set up information for the solver arguments
            cuuplo = cublasfill(uplo)
            n = checksquare(A[1])
            lda = max(1, stride(A[1], 2))
            batchSize = length(A)
            devinfo = CuArray{Cint}(undef, batchSize)

            Aptrs = unsafe_batch(A)

            # Run the solver
            $fname(dense_handle(), cuuplo, n, Aptrs, lda, devinfo, batchSize)

            # Copy the solver info and delete the device memory
            info = @allowscalar collect(devinfo)
            unsafe_free!(devinfo)

            # Double check the solver's exit status
            for i = 1:batchSize
                if info[i] < 0
                    throw(ArgumentError("The $(info)th parameter of the $(i)th solver is wrong"))
                end
            end

            # info[i] > 0 means the leading minor of order info[i] is not positive definite
            # LinearAlgebra.LAPACK does not throw Exception here
            # to simplify calls to isposdef! and factorize
            return A, info
        end
    end
end
