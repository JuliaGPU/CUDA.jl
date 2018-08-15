
#potrf 
for (bname, fname,elty) in ((:cusolverDnSpotrf_bufferSize, :cusolverDnSpotrf, :Float32),
                            (:cusolverDnDpotrf_bufferSize, :cusolverDnDpotrf, :Float64),
                            (:cusolverDnCpotrf_bufferSize, :cusolverDnCpotrf, :ComplexF32),
                            (:cusolverDnZpotrf_bufferSize, :cusolverDnZpotrf, :ComplexF64))
    @eval begin
        function potrf!(uplo::Char,
                        A::CuMatrix{$elty})
            cuuplo = cublasfill(uplo)
            n = size(A, 1)
            if size(A, 2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            lda     = max(1, stride(A, 2))
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, cublasFillMode_t, Cint,
                          Ptr{$elty}, Cint, Ref{Cint}),
                         libcusolver_handle_dense[], cuuplo, n, A, lda, bufSize)

            buffer  = CuArray{$elty}(bufSize[])
            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, cublasFillMode_t, Cint,
                          Ptr{$elty}, Cint, Ptr{$elty}, Cint, Ptr{Cint}),
                         libcusolver_handle_dense[], cuuplo, n, A, lda, buffer,
                         bufSize[], devinfo)
            info = _getindex(devinfo, 1)
            if info < 0
                throw(ArgumentError("The $(-info)th parameter is wrong"))
            elseif info > 0
                throw(LinearAlgebra.SingularException(info))
            end
            A
        end
    end
end

#getrf 
for (bname, fname,elty) in ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
                            (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64),
                            (:cusolverDnCgetrf_bufferSize, :cusolverDnCgetrf, :ComplexF32),
                            (:cusolverDnZgetrf_bufferSize, :cusolverDnZgetrf, :ComplexF64))
    @eval begin
        function getrf!(A::CuMatrix{$elty})
            m,n     = size(A)
            lda     = max(1, stride(A, 2))
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ptr{$elty}, Cint,
                          Ref{Cint}), libcusolver_handle_dense[], m, n, A, lda,
                         bufSize)

            buffer  = CuArray{$elty}(bufSize[])
            devipiv = CuArray{Cint}(min(m,n))
            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ptr{$elty},
                          Cint, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                         libcusolver_handle_dense[], m, n, A, lda, buffer,
                         devipiv, devinfo)
            info = _getindex(devinfo, 1)
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
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ptr{$elty}, Cint,
                          Ref{Cint}), libcusolver_handle_dense[], m, n, A,
                         lda, bufSize)
            buffer  = CuArray{$elty}(bufSize[])
            tau  = CuArray{$elty}(min(m, n))
            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ptr{$elty},
                          Cint, Ptr{$elty}, Ptr{$elty}, Cint, Ptr{Cint}),
                         libcusolver_handle_dense[], m, n, A, lda, tau, buffer,
                         bufSize[], devinfo)
            info = _getindex(devinfo, 1)
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
            n      = size(A, 1)
            if size(A,2) != n
                throw(DimensionMismatch("SymTridiagonal matrix must be square!"))
            end
            lda = max(1, stride(A, 2))
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Ptr{$elty}, Cint,
                          Ref{Cint}), libcusolver_handle_dense[], n, A, lda,
                         bufSize)

            buffer  = CuArray{$elty}(bufSize[])
            devipiv = CuArray{Cint}(n)
            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, cublasFillMode_t, Cint,
                          Ptr{$elty}, Cint, Ptr{Cint}, Ptr{$elty}, Cint,
                          Ptr{Cint}), libcusolver_handle_dense[], cuuplo, n, A,
                         lda, devipiv, buffer, bufSize[], devinfo)
            info = _getindex(devinfo, 1)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            elseif info > 0
                throw(LinearAlgebra.SingularException(info))
            end
            A, devipiv
        end
    end
end

#potrs
for (fname,elty) in ((:cusolverDnSpotrs, :Float32),
                     (:cusolverDnDpotrs, :Float64),
                     (:cusolverDnCpotrs, :ComplexF32),
                     (:cusolverDnZpotrs, :ComplexF64))
    @eval begin
        function potrs!(uplo::Char,
                        A::CuMatrix{$elty},
                        B::CuVecOrMat{$elty})
            cuuplo = cublasfill(uplo)
            n = size(A, 1)
            if size(A, 2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B,2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))

            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                          Ptr{$elty}, Cint, Ptr{$elty}, Cint, Ptr{Cint}),
                         libcusolver_handle_dense[], cuuplo, n, nrhs, A, lda, B,
                         ldb, devinfo)
            info = _getindex(devinfo, 1)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end
            B
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

            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, cublasOperation_t, Cint, Cint,
                          Ptr{$elty}, Cint, Ptr{Cint}, Ptr{$elty}, Cint,
                          Ptr{Cint}), libcusolver_handle_dense[], cutrans, n, nrhs,
                         A, lda, ipiv, B, ldb, devinfo)
            info = _getindex(devinfo, 1)
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
                             (:cusolverDnZunmqr_bufferSize, :cusolverDnZunmqr, :ComplexF64))    @eval begin
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
                    Ctemp = CuArray{$elty}(m - ldc, n) .= 0
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
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                          (cusolverDnHandle_t, cublasSideMode_t,
                           cublasOperation_t, Cint, Cint, Cint, Ptr{$elty}, 
                           Cint, Ptr{$elty}, Ptr{$elty}, Cint, Ref{Cint}),
                          libcusolver_handle_dense[], cuside,
                          cutrans, m, n, k, A,
                          lda, tau, C, ldc, bufSize)
            buffer  = CuArray{$elty}(bufSize[])
            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, cublasSideMode_t,
                          cublasOperation_t, Cint, Cint, Cint, Ptr{$elty},
                          Cint, Ptr{$elty}, Ptr{$elty}, Cint, Ptr{$elty},
                          Cint, Ptr{Cint}),
                         libcusolver_handle_dense[], cuside,
                         cutrans, m, n, k, A, lda, tau, C, ldc, buffer,
                         bufSize[], devinfo)
            info = _getindex(devinfo, 1)
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
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)), libcusolver), cusolverStatus_t,
                          (cusolverDnHandle_t, Cint, Cint, Cint, Ptr{$elty}, Cint,
                           Ptr{$elty}, Ref{Cint}),
                          libcusolver_handle_dense[], m, n, k, A, lda, tau, bufSize)
            buffer  = CuArray{$elty}(bufSize[])
            devinfo = CuArray{Cint}(1)
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Cint, Ptr{$elty},
                          Cint, Ptr{$elty}, Ptr{$elty}, Cint, Ptr{Cint}),
                         libcusolver_handle_dense[], m, n, k, A,
                         lda, tau, buffer, bufSize[], devinfo)
            info = _getindex(devinfo, 1)
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
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ref{Cint}),
                        libcusolver_handle_dense[], m, n, bufSize)
            buffer  = CuArray{$elty}(bufSize[])
            devinfo = CuArray{Cint}(1)
            k       = min(m, n)
            D       = CuArray{$relty}(k)
            E       = CuArray{$relty}(k) .= 0
            TAUQ    = CuArray{$elty}(k)
            TAUP    = CuArray{$elty}(k)
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ptr{$elty},
                          Cint, Ptr{$relty}, Ptr{$relty}, Ptr{$elty},
                          Ptr{$elty}, Ptr{$elty}, Cint, Ptr{Cint}),
                         libcusolver_handle_dense[], m, n, A, lda, D, E, TAUQ,
                         TAUP, buffer, bufSize[], devinfo)
            info = _getindex(devinfo, 1)
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
            m,n     = size(A)
            lda     = max(1, stride(A, 2))
            bufSize = Ref{Cint}(0)
            @check ccall(($(string(bname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cint, Cint, Ref{Cint}),
                        libcusolver_handle_dense[], m, n, bufSize)

            buffer  = CuArray{$elty}(bufSize[])
            rbuffer = CuArray{$relty}(5 * min(m, n))
            devinfo = CuArray{Cint}(1)
            U       = CuArray{$elty}(m, m)
            ldu     = max(1, stride(U, 2))
            S       = CuArray{$relty}(min(m, n))
            Vt      = CuArray{$elty}(n, n)
            ldvt    = max(1, stride(Vt, 2))
            @check ccall(($(string(fname)), libcusolver), cusolverStatus_t,
                         (cusolverDnHandle_t, Cuchar, Cuchar, Cint,
                          Cint, Ptr{$elty}, Cint, Ptr{$relty}, Ptr{$elty},
                          Cint, Ptr{$elty}, Cint, Ptr{$elty}, Cint,
                          Ptr{$relty}, Ptr{Cint}), libcusolver_handle_dense[],
                         jobu, jobvt, m, n, A, lda, S, U, ldu, Vt, ldvt,
                         buffer, bufSize[], rbuffer, devinfo)
            info = _getindex(devinfo, 1)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end
            U, S, Vt
        end
    end
end
