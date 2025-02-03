
using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using ..CUBLAS: unsafe_batch

function cusolverDnCreate()
  handle_ref = Ref{cusolverDnHandle_t}()
  cusolverDnCreate(handle_ref)
  return handle_ref[]
end

# po
## potrf
for (bname, fname,elty) in ((:cusolverDnSpotrf_bufferSize, :cusolverDnSpotrf, :Float32),
                            (:cusolverDnDpotrf_bufferSize, :cusolverDnDpotrf, :Float64),
                            (:cusolverDnCpotrf_bufferSize, :cusolverDnCpotrf, :ComplexF32),
                            (:cusolverDnZpotrf_bufferSize, :cusolverDnZpotrf, :ComplexF64))
    @eval begin
        function potrf!(uplo::Char,
                        A::StridedCuMatrix{$elty})
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, uplo, n, A, lda, out)
                out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, uplo, n, A, lda,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
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
        function potrs!(uplo::Char,
                        A::StridedCuMatrix{$elty},
                        B::StridedCuVecOrMat{$elty})
            chkuplo(uplo)
            n = checksquare(A)
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B,2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))
            dh = dense_handle()

            $fname(dh, uplo, n, nrhs, A, lda, B, ldb, dh.info)

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            B
        end
    end
end

for (bname, fname,elty) in ((:cusolverDnSpotri_bufferSize, :cusolverDnSpotri, :Float32),
                            (:cusolverDnDpotri_bufferSize, :cusolverDnDpotri, :Float64),
                            (:cusolverDnCpotri_bufferSize, :cusolverDnCpotri, :ComplexF32),
                            (:cusolverDnZpotri_bufferSize, :cusolverDnZpotri, :ComplexF64))
    @eval begin
        function potri!(uplo::Char,
                        A::StridedCuMatrix{$elty})
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, uplo, n, A, lda, out)
                out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, uplo, n, A, lda,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
        function getrf!(A::StridedCuMatrix{$elty}, ipiv::CuVector{Cint})
            m,n = size(A)
            lda = max(1, stride(A, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, m, n, A, lda, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, m, n, A, lda, buffer, ipiv, dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            A, ipiv, info
        end

        function getrf!(A::StridedCuMatrix{$elty})
            m,n = size(A)
            ipiv = CuArray{Cint}(undef, min(m, n))
            getrf!(A, ipiv)
        end
    end
end

#geqrf
for (bname, fname,elty) in ((:cusolverDnSgeqrf_bufferSize, :cusolverDnSgeqrf, :Float32),
                            (:cusolverDnDgeqrf_bufferSize, :cusolverDnDgeqrf, :Float64),
                            (:cusolverDnCgeqrf_bufferSize, :cusolverDnCgeqrf, :ComplexF32),
                            (:cusolverDnZgeqrf_bufferSize, :cusolverDnZgeqrf, :ComplexF64))
    @eval begin
        function geqrf!(A::StridedCuMatrix{$elty}, tau::CuVector{$elty})
            m, n = size(A)
            lda  = max(1, stride(A, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, m, n, A, lda, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, m, n, A, lda, tau,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            A, tau
        end

        function geqrf!(A::StridedCuMatrix{$elty})
            m, n = size(A)
            tau = similar(A, $elty, min(m,n))
            geqrf!(A, tau)
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
                        A::StridedCuMatrix{$elty},
                        ipiv::CuVector{Cint})
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, n, A, lda, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, uplo, n, A, lda, ipiv,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            A, ipiv, info
        end

        function sytrf!(uplo::Char, A::StridedCuMatrix{$elty}; pivoting::Bool=true)
            n = checksquare(A)
            if pivoting
                ipiv = CuArray{Cint}(undef, n)
                return sytrf!(uplo, A, ipiv)
            else
                chkuplo(uplo)
                n = checksquare(A)
                lda = max(1, stride(A, 2))
                dh = dense_handle()

                function bufferSize()
                    out = Ref{Cint}(0)
                    $bname(dh, n, A, lda, out)
                    return out[] * sizeof($elty)
                end

                with_workspace(dh.workspace_gpu, bufferSize) do buffer
                    $fname(dh, uplo, n, A, lda, CU_NULL,
                           buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
                end

                info = @allowscalar dh.info[1]
                chkargsok(info |> BlasInt)

                return A, info
            end
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
                        A::StridedCuMatrix{$elty},
                        ipiv::CuVector{Cint},
                        B::StridedCuVecOrMat{$elty})

            # Support transa = 'C' for real matrices
            trans = $elty <: Real && trans == 'C' ? 'T' : trans

            chktrans(trans)
            n = checksquare(A)
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match dimension of A, $n"))
            end
            if length(ipiv) != n
                throw(DimensionMismatch("length of ipiv, $(length(ipiv)), must match dimension of A, $n"))
            end
            nrhs = size(B, 2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))
            dh = dense_handle()

            $fname(dh, trans, n, nrhs, A, lda, ipiv, B, ldb, dh.info)

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
                        A::StridedCuMatrix{$elty},
                        tau::CuVector{$elty},
                        C::StridedCuVecOrMat{$elty})

            # Support transa = 'C' for real matrices
            trans = $elty <: Real && trans == 'C' ? 'T' : trans
           ($elty <: Complex) && (trans == 'T') && throw(ArgumentError("trans = 'T' is not supported with complex matrices."))

            chkside(side)
            chktrans(trans)
            m,n = ndims(C) == 2 ? size(C) : (length(C), 1)
            mA  = size(A, 1)
            k   = length(tau)
            if side == 'L' && m != mA
                throw(DimensionMismatch("for a left-sided multiplication, the first dimension of C, $m, must equal the second dimension of A, $mA"))
            end
            if side == 'R' && n != mA
                throw(DimensionMismatch("for a right-sided multiplication, the second dimension of C, $m, must equal the second dimension of A, $mA"))
            end
            if side == 'L' && k > m
                throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= m = $m"))
            end
            if side == 'R' && k > n
                throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= n = $n"))
            end
            lda = max(1, stride(A, 2))
            ldc = max(1, stride(C, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, side, trans, m, n, k, A, lda, tau, C, ldc, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, side, trans, m, n, k, A, lda, tau, C, ldc,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            C
        end
    end
end

#orgqr
for (bname, fname, elty) in ((:cusolverDnSorgqr_bufferSize, :cusolverDnSorgqr, :Float32),
                             (:cusolverDnDorgqr_bufferSize, :cusolverDnDorgqr, :Float64),
                             (:cusolverDnCungqr_bufferSize, :cusolverDnCungqr, :ComplexF32),
                             (:cusolverDnZungqr_bufferSize, :cusolverDnZungqr, :ComplexF64))
    @eval begin
        function orgqr!(A::StridedCuMatrix{$elty}, tau::CuVector{$elty})
            m = size(A , 1)
            n = min(m, size(A, 2))
            lda = max(1, stride(A, 2))
            k = length(tau)
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, m, n, k, A, lda, tau, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, m, n, k, A, lda, tau,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
        function gebrd!(A::StridedCuMatrix{$elty})
            m, n = size(A)
            lda  = max(1, stride(A, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, m, n, out)
                return out[] * sizeof($elty)
            end

            k       = min(m, n)
            D       = similar(A, $relty, k)
            E       = fill!(similar(A, $relty, k), zero($relty))
            TAUQ    = similar(A, $elty, k)
            TAUP    = similar(A, $elty, k)

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, m, n, A, lda, D, E, TAUQ, TAUP,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
                        A::StridedCuMatrix{$elty})
            m, n = size(A)
            (m < n) && throw(ArgumentError("CUSOLVER's gesvd requires m ≥ n"))
            k = min(m, n)
            lda = max(1, stride(A, 2))

            U = if jobu === 'A'
                similar(A, $elty, (m, m))
            elseif jobu === 'S'
                similar(A, $elty, (m, k))
            elseif jobu === 'N' || jobu === 'O'
                CU_NULL
            else
                error("jobu must be one of 'A', 'S', 'O', or 'N'")
            end
            ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
            S = similar(A, $relty, k)
            Vt = if jobvt === 'A'
                similar(A, $elty, (n, n))
            elseif jobvt === 'S'
                similar(A, $elty, (k, n))
            elseif jobvt === 'N' || jobvt === 'O'
                CU_NULL
            else
                error("jobvt must be one of 'A', 'S', 'O', or 'N'")
            end
            ldvt = Vt == CU_NULL ? 1 : max(1, stride(Vt, 2))
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, m, n, out)
                return out[] * sizeof($elty)
            end

            rwork   = CuArray{$relty}(undef, min(m, n) - 1)
            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, jobu, jobvt, m, n, A, lda, S, U, ldu, Vt, ldvt,
                       buffer, sizeof(buffer) ÷ sizeof($elty), rwork, dh.info)
            end
            unsafe_free!(rwork)

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
                         A::StridedCuMatrix{$elty};
                         tol::$relty=eps($relty),
                         max_sweeps::Int=100)
            m,n     = size(A)
            lda     = max(1, stride(A, 2))
            # Warning! For some reason, the solver needs to access U and V even
            # when only the values are requested
            U = if jobz === 'V' && econ == 1 && m > n
                similar(A, $elty, (m ,n))
            else
                similar(A, $elty, (m, m))
            end
            ldu     = max(1, stride(U, 2))

            S       = similar(A, $relty, min(m,n))

            V = if jobz === 'V' && econ == 1 && m < n
                similar(A, $elty, (n, m))
            else
                similar(A, $elty, (n, n))
            end
            ldv     = max(1, stride(V, 2))

            params  = Ref{gesvdjInfo_t}(C_NULL)
            cusolverDnCreateGesvdjInfo(params)
            cusolverDnXgesvdjSetTolerance(params[], tol)
            cusolverDnXgesvdjSetMaxSweeps(params[], max_sweeps)
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                       out, params[])
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info, params[])
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            cusolverDnDestroyGesvdjInfo(params[])

            U, S, V
        end
    end
end

for (bname, fname, elty, relty) in ((:cusolverDnSgesvdjBatched_bufferSize, :cusolverDnSgesvdjBatched, :Float32, :Float32),
                                    (:cusolverDnDgesvdjBatched_bufferSize, :cusolverDnDgesvdjBatched, :Float64, :Float64),
                                    (:cusolverDnCgesvdjBatched_bufferSize, :cusolverDnCgesvdjBatched, :ComplexF32, :Float32),
                                    (:cusolverDnZgesvdjBatched_bufferSize, :cusolverDnZgesvdjBatched, :ComplexF64, :Float64))
    @eval begin
        function gesvdj!(jobz::Char,
                         A::StridedCuArray{$elty,3};
                         tol::$relty=eps($relty),
                         max_sweeps::Int=100)
            m, n, batchSize = size(A)
            if m > 32 || n > 32
                throw(ArgumentError("CUSOLVER's gesvdjBatched currently requires m <=32 and n <= 32"))
            end
            lda = max(1, stride(A, 2))
            
            U = similar(A, $elty, (m, m, batchSize))
            ldu = max(1, stride(U, 2))

            S = similar(A, $relty, (min(m, n), batchSize))

            V = similar(A, $elty, n, n, batchSize)
            ldv = max(1, stride(V, 2))

            params = Ref{gesvdjInfo_t}(C_NULL)
            cusolverDnCreateGesvdjInfo(params)
            cusolverDnXgesvdjSetTolerance(params[], tol)
            cusolverDnXgesvdjSetMaxSweeps(params[], max_sweeps)

            dh = dense_handle()
            resize!(dh.info, batchSize)

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                       out, params[], batchSize)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info, params[], batchSize)
            end

            info = @allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(BlasInt(info[i]))
            end

            cusolverDnDestroyGesvdjInfo(params[])

            U, S, V
        end
    end
end

for (bname, fname, elty, relty) in ((:cusolverDnSgesvdaStridedBatched_bufferSize, :cusolverDnSgesvdaStridedBatched, :Float32, :Float32),
                                    (:cusolverDnDgesvdaStridedBatched_bufferSize, :cusolverDnDgesvdaStridedBatched, :Float64, :Float64),
                                    (:cusolverDnCgesvdaStridedBatched_bufferSize, :cusolverDnCgesvdaStridedBatched, :ComplexF32, :Float32),
                                    (:cusolverDnZgesvdaStridedBatched_bufferSize, :cusolverDnZgesvdaStridedBatched, :ComplexF64, :Float64))
    @eval begin
        function gesvda!(jobz::Char,
                         A::StridedCuArray{$elty,3};
                         rank::Int=min(size(A,1), size(A,2)))
            m, n, batchSize = size(A)
            if m < n
                throw(ArgumentError("CUSOLVER's gesvda currently requires m >= n"))
                # nikopj: I can't find the documentation for this...
            end
            lda = max(1, stride(A, 2))
            strideA = stride(A, 3)

            U = similar(A, $elty, (m, rank, batchSize))
            ldu = max(1, stride(U, 2))
            strideU = stride(U, 3)

            S = similar(A, $relty, (rank, batchSize))
            strideS = stride(S, 2)

            V = similar(A, $elty, (n, rank, batchSize))
            ldv = max(1, stride(V, 2))
            strideV = stride(V, 3)

            dh = dense_handle()
            resize!(dh.info, batchSize)

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, jobz, rank, m, n, A, lda, strideA,
                       S, strideS, U, ldu, strideU, V, ldv, strideV,
                       out, batchSize)
                return out[] * sizeof($elty)
            end

            # residual storage
            h_RnrmF = Array{Cdouble}(undef, batchSize)

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, jobz, rank, m, n, A, lda, strideA,
                       S, strideS, U, ldu, strideU, V, ldv, strideV,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info, h_RnrmF, batchSize)
            end

            info = @allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(BlasInt(info[i]))
            end

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
                        A::StridedCuMatrix{$elty})
            chkuplo(uplo)
            n   = checksquare(A)
            lda = max(1, stride(A, 2))
            W   = CuArray{$relty}(undef, n)
            dh  = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, jobz, uplo, n, A, lda, W, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, jobz, uplo, n, A, lda, W,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
                        A::StridedCuMatrix{$elty},
                        B::StridedCuMatrix{$elty})
            chkuplo(uplo)
            nA, nB  = checksquare(A, B)
            if nB != nA
                throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
            end
            n   = nA
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            W   = CuArray{$relty}(undef, n)
            dh  = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W, out)
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
                        A::StridedCuMatrix{$elty},
                        B::StridedCuMatrix{$elty};
                        tol::$relty=eps($relty),
                        max_sweeps::Int=100)
            chkuplo(uplo)
            nA, nB  = checksquare(A, B)
            if nB != nA
                throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
            end
            n      = nA
            lda    = max(1, stride(A, 2))
            ldb    = max(1, stride(B, 2))
            W      = CuArray{$relty}(undef, n)
            params = Ref{syevjInfo_t}(C_NULL)
            cusolverDnCreateSyevjInfo(params)
            cusolverDnXsyevjSetTolerance(params[], tol)
            cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)
            dh = dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
                       out, params[])
                return out[] * sizeof($elty)
            end

            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
                       buffer, sizeof(buffer) ÷ sizeof($elty), dh.info, params[])
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

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
                                           (:heevjBatched!, :cusolverDnZheevjBatched_bufferSize, :cusolverDnZheevjBatched, :ComplexF64, :Float64))
    @eval begin
        function $jname(jobz::Char,
                        uplo::Char,
                        A::StridedCuArray{$elty};
                        tol::$relty=eps($relty),
                        max_sweeps::Int=100)

            # Set up information for the solver arguments
            chkuplo(uplo)
            n         = checksquare(A)
            lda       = max(1, stride(A, 2))
            batchSize = size(A,3)
            W         = CuArray{$relty}(undef, n,batchSize)
            params    = Ref{syevjInfo_t}(C_NULL)

            dh = dense_handle()
            resize!(dh.info, batchSize)

            # Initialize the solver parameters
            cusolverDnCreateSyevjInfo(params)
            cusolverDnXsyevjSetTolerance(params[], tol)
            cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)

            # Calculate the workspace size
            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, jobz, uplo, n, A, lda, W, out, params[], batchSize)
                return out[] * sizeof($elty)
            end

            # Run the solver
            with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(dh, jobz, uplo, n, A, lda, W, buffer,
                       sizeof(buffer) ÷ sizeof($elty), dh.info, params[], batchSize)
            end

            # Copy the solver info and delete the device memory
            info = @allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(BlasInt(info[i]))
            end

            cusolverDnDestroySyevjInfo(params[])

            # Return eigenvalues (in W) and possibly eigenvectors (in A)
            if jobz == 'N'
                return W
            elseif jobz == 'V'
                return W, A
            end
        end
    end
end

for (fname, elty) in ((:cusolverDnSpotrsBatched, :Float32),
                      (:cusolverDnDpotrsBatched, :Float64),
                      (:cusolverDnCpotrsBatched, :ComplexF32),
                      (:cusolverDnZpotrsBatched, :ComplexF64))
    @eval begin
        function potrsBatched!(uplo::Char,
                               A::Vector{<:StridedCuMatrix{$elty}},
                               B::Vector{<:StridedCuVecOrMat{$elty}})
            if length(A) != length(B)
                throw(DimensionMismatch(""))
            end
            # Set up information for the solver arguments
            chkuplo(uplo)
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

            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)

            dh = dense_handle()

            # Run the solver
            $fname(dh, uplo, n, nrhs, Aptrs, lda, Bptrs, ldb, dh.info, batchSize)

            # Copy the solver info and delete the device memory
            info = @allowscalar dh.info[1]
            chklapackerror(BlasInt(info))

            return B
        end
    end
end

for (fname, elty) in ((:cusolverDnSpotrfBatched, :Float32),
                      (:cusolverDnDpotrfBatched, :Float64),
                      (:cusolverDnCpotrfBatched, :ComplexF32),
                      (:cusolverDnZpotrfBatched, :ComplexF64))
    @eval begin
        function potrfBatched!(uplo::Char, A::Vector{<:StridedCuMatrix{$elty}})

            # Set up information for the solver arguments
            chkuplo(uplo)
            n = checksquare(A[1])
            lda = max(1, stride(A[1], 2))
            batchSize = length(A)

            Aptrs = unsafe_batch(A)

            dh = dense_handle()
            resize!(dh.info, batchSize)

            # Run the solver
            $fname(dh, uplo, n, Aptrs, lda, dh.info, batchSize)

            # Copy the solver info and delete the device memory
            info = @allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(BlasInt(info[i]))
            end

            # info[i] > 0 means the leading minor of order info[i] is not positive definite
            # LinearAlgebra.LAPACK does not throw Exception here
            # to simplify calls to isposdef! and factorize
            return A, info
        end
    end
end

# gesv
function gesv!(X::CuVecOrMat{T}, A::CuMatrix{T}, B::CuVecOrMat{T}; fallback::Bool=true,
               residual_history::Bool=false, irs_precision::String="AUTO", refinement_solver::String="CLASSICAL",
               maxiters::Int=0, maxiters_inner::Int=0, tol::Float64=0.0, tol_inner=Float64=0.0) where T <: BlasFloat

    params = CuSolverIRSParameters()
    info = CuSolverIRSInformation()
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldx = max(1, stride(X, 2))
    niters = Ref{Cint}()
    dh = dense_handle()

    if irs_precision == "AUTO"
        (T == Float32)    && (irs_precision = "R_32F")
        (T == Float64)    && (irs_precision = "R_64F")
        (T == ComplexF32) && (irs_precision = "C_32F")
        (T == ComplexF64) && (irs_precision = "C_64F")
    else
        (T == Float32)    && (irs_precision ∈ ("R_32F", "R_16F", "R_16BF", "R_TF32") || error("$irs_precision is not supported."))
        (T == Float64)    && (irs_precision ∈ ("R_64F", "R_32F", "R_16F", "R_16BF", "R_TF32") || error("$irs_precision is not supported."))
        (T == ComplexF32) && (irs_precision ∈ ("C_32F", "C_16F", "C_16BF", "C_TF32") || error("$irs_precision is not supported."))
        (T == ComplexF64) && (irs_precision ∈ ("C_64F", "C_32F", "C_16F", "C_16BF", "C_TF32") || error("$irs_precision is not supported."))
    end
    cusolverDnIRSParamsSetSolverMainPrecision(params, T)
    cusolverDnIRSParamsSetSolverLowestPrecision(params, irs_precision)
    cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    (tol != 0.0) && cusolverDnIRSParamsSetTol(params, tol)
    (tol_inner != 0.0) && cusolverDnIRSParamsSetTolInner(params, tol_inner)
    (maxiters != 0) && cusolverDnIRSParamsSetMaxIters(params, maxiters)
    (maxiters_inner != 0) && cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    fallback ? cusolverDnIRSParamsEnableFallback(params) : cusolverDnIRSParamsDisableFallback(params)
    residual_history && cusolverDnIRSInfosRequestResidual(info)

    function bufferSize()
        buffer_size = Ref{Csize_t}(0)
        cusolverDnIRSXgesv_bufferSize(dh, params, n, nrhs, buffer_size)
        return buffer_size[]
    end

    with_workspace(dh.workspace_gpu, bufferSize) do buffer
        cusolverDnIRSXgesv(dh, params, info, n, nrhs, A, lda, B, ldb,
                           X, ldx, buffer, sizeof(buffer), niters, dh.info)
    end

    # Copy the solver flag and delete the device memory
    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)

    return X, info
end

# gels
function gels!(X::CuVecOrMat{T}, A::CuMatrix{T}, B::CuVecOrMat{T}; fallback::Bool=true,
               residual_history::Bool=false, irs_precision::String="AUTO", refinement_solver::String="CLASSICAL",
               maxiters::Int=0, maxiters_inner::Int=0, tol::Float64=0.0, tol_inner=Float64=0.0) where T <: BlasFloat

    params = CuSolverIRSParameters()
    info = CuSolverIRSInformation()
    m,n = size(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldx = max(1, stride(X, 2))
    niters = Ref{Cint}()
    dh = dense_handle()

    if irs_precision == "AUTO"
        (T == Float32)    && (irs_precision = "R_32F")
        (T == Float64)    && (irs_precision = "R_64F")
        (T == ComplexF32) && (irs_precision = "C_32F")
        (T == ComplexF64) && (irs_precision = "C_64F")
    else
        (T == Float32)    && (irs_precision ∈ ("R_32F", "R_16F", "R_16BF", "R_TF32") || error("$irs_precision is not supported."))
        (T == Float64)    && (irs_precision ∈ ("R_64F", "R_32F", "R_16F", "R_16BF", "R_TF32") || error("$irs_precision is not supported."))
        (T == ComplexF32) && (irs_precision ∈ ("C_32F", "C_16F", "C_16BF", "C_TF32") || error("$irs_precision is not supported."))
        (T == ComplexF64) && (irs_precision ∈ ("C_64F", "C_32F", "C_16F", "C_16BF", "C_TF32") || error("$irs_precision is not supported."))
    end
    cusolverDnIRSParamsSetSolverMainPrecision(params, T)
    cusolverDnIRSParamsSetSolverLowestPrecision(params, irs_precision)
    cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    (tol != 0.0) && cusolverDnIRSParamsSetTol(params, tol)
    (tol_inner != 0.0) && cusolverDnIRSParamsSetTolInner(params, tol_inner)
    (maxiters != 0) && cusolverDnIRSParamsSetMaxIters(params, maxiters)
    (maxiters_inner != 0) && cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    fallback ? cusolverDnIRSParamsEnableFallback(params) : cusolverDnIRSParamsDisableFallback(params)
    residual_history && cusolverDnIRSInfosRequestResidual(info)

    function bufferSize()
        buffer_size = Ref{Csize_t}(0)
        cusolverDnIRSXgels_bufferSize(dh, params, m, n, nrhs, buffer_size)
        return buffer_size[]
    end

    with_workspace(dh.workspace_gpu, bufferSize) do buffer
        cusolverDnIRSXgels(dh, params, info, m, n, nrhs, A, lda, B, ldb,
                           X, ldx, buffer, sizeof(buffer), niters, dh.info)
    end

    # Copy the solver flag and delete the device memory
    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)

    return X, info
end

# LAPACK
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        LAPACK.potrf!(uplo::Char, A::StridedCuMatrix{$elty}) = potrf!(uplo, A)
        LAPACK.potrs!(uplo::Char, A::StridedCuMatrix{$elty}, B::StridedCuVecOrMat{$elty}) = potrs!(uplo, A, B)
        LAPACK.potri!(uplo::Char, A::StridedCuMatrix{$elty}) = potri!(uplo, A)
        LAPACK.getrf!(A::StridedCuMatrix{$elty}) = getrf!(A)
        LAPACK.getrf!(A::StridedCuMatrix{$elty}, ipiv::CuVector{Cint}) = getrf!(A, ipiv)
        LAPACK.geqrf!(A::StridedCuMatrix{$elty}) = geqrf!(A)
        LAPACK.geqrf!(A::StridedCuMatrix{$elty}, tau::CuVector{$elty}) = geqrf!(A, tau)
        LAPACK.sytrf!(uplo::Char, A::StridedCuMatrix{$elty}) = sytrf!(uplo, A)
        LAPACK.sytrf!(uplo::Char, A::StridedCuMatrix{$elty}, ipiv::CuVector{Cint}) = sytrf!(uplo, A, ipiv)
        LAPACK.getrs!(trans::Char, A::StridedCuMatrix{$elty}, ipiv::CuVector{Cint}, B::StridedCuVecOrMat{$elty}) = getrs!(trans, A, ipiv, B)
        LAPACK.ormqr!(side::Char, trans::Char, A::CuMatrix{$elty}, tau::CuVector{$elty}, C::CuVecOrMat{$elty}) = ormqr!(side, trans, A, tau, C)
        LAPACK.orgqr!(A::StridedCuMatrix{$elty}, tau::CuVector{$elty}) = orgqr!(A, tau)
        LAPACK.gebrd!(A::StridedCuMatrix{$elty}) = gebrd!(A)
        LAPACK.gesvd!(jobu::Char, jobvt::Char, A::StridedCuMatrix{$elty}) = gesvd!(jobu, jobvt, A)
    end
end

for elty in (:Float32, :Float64)
    @eval begin
        LAPACK.syev!(jobz::Char, uplo::Char, A::StridedCuMatrix{$elty}) = syevd!(jobz, uplo, A)
        LAPACK.sygvd!(itype::Int, jobz::Char, uplo::Char, A::StridedCuMatrix{$elty}, B::StridedCuMatrix{$elty}) = sygvd!(itype, jobz, uplo, A, B)
    end
end

for elty in (:ComplexF32, :ComplexF64)
    @eval begin
        LAPACK.syev!(jobz::Char, uplo::Char, A::StridedCuMatrix{$elty}) = heevd!(jobz, uplo, A)
        LAPACK.sygvd!(itype::Int, jobz::Char, uplo::Char, A::StridedCuMatrix{$elty}, B::StridedCuMatrix{$elty}) = hegvd!(itype, jobz, uplo, A, B)
    end
end

for elty in (:Float32, :Float64)
    @eval begin
        LAPACK.syevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{$elty}) = syevd!(jobz, uplo, A)
    end
end

for elty in (:ComplexF32, :ComplexF64)
    @eval begin
        LAPACK.syevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{$elty}) = heevd!(jobz, uplo, A)
    end
end
