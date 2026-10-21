
using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using cuBLAS: unsafe_batch

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
                cuSOLVER.version() < v"11.7.2" && throw(ErrorException("This operation is not supported by the current CUDA version."))
                # NVIDIA bug #5949478: cusolverDn{C,Z}sytrf with NULL ipiv
                # performs Hermitian (LDL^H) factorization instead of symmetric
                # (LDL^T) on CUSOLVER 12.0.9 (CUDA 13.1), silently producing
                # wrong results for complex symmetric matrices.
                if $elty <: Complex && v"12.0.9" <= cuSOLVER.version() < v"12.1"
                    throw(ErrorException("Non-pivoting symmetric factorization of complex matrices is broken on cuSOLVER $(cuSOLVER.version()). Use pivoting=true instead."))
                end
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

# Blocked Householder (compact WY) application of Q
#
# cuSOLVER only offers ormqr and orgqr in its legacy API, which reports the workspace size
# as a 32-bit element count. That workspace grows with the number of rows, by an amount
# that differs between cuSOLVER versions, so these routines reject (or, for orgqr, silently
# under-allocate for) matrices with millions of rows, however much memory is available
# (JuliaGPU/CUDA.jl#3234). For those problems we apply the reflectors ourselves, a panel at
# a time, using only cuBLAS. cuSOLVER's Xlarft is avoided: its workspace is as large as the
# panel, which for tall and skinny matrices means another copy of A.
#
# A panel of reflectors H(1)⋯H(k) = I - V T Vᴴ, where V is unit lower trapezoidal and is
# read in place from the output of geqrf. The triangular factor is T = R⁻¹ D, with
# D = diag(τ) and R = I + D triu(VᴴV, 1) unit upper triangular (the UT transform of
# Joffrain et al., "Accumulating Householder transformations, revisited", 2006, scaled so
# that τ = 0 needs no special case). We keep R and multiply by T with a triangular solve.

const HOUSEHOLDER_BLOCKSIZE = 128

# whether to prefer the blocked implementation even when cuSOLVER accepts the problem. For
# tall matrices the legacy routines' workspace grows with the number of rows rather than
# with the number of reflectors (to about 5x the size of A for k = 64), and they are slower:
# on an RTX 5080 the blocked ormqr is 1.5-4x faster from 2^15 rows and m >= 8k, while orgqr
# is up to 3.7x faster for small k and on par otherwise. Smaller problems are dominated by
# the blocked implementation's launch overhead.
prefer_blocked(m, k) = m >= max(2^15, 8k)

# compute R for the panel V with reflector coefficients τ. V1 receives the leading jb×jb
# block of V in explicit unit lower triangular form.
function householder_ut!(R, V1, V, tau)
    T = eltype(V)
    mv, jb = size(V)
    transv = T <: Real ? 'T' : 'C'
    V1 .= ifelse.((1:jb) .> (1:jb)', view(V, 1:jb, :), T.((1:jb) .== (1:jb)'))
    cuBLAS.gemm!(transv, 'N', one(T), V1, V1, zero(T), R)
    if mv > jb
        V2 = view(V, (jb+1):mv, :)
        cuBLAS.gemm!(transv, 'N', one(T), V2, V2, one(T), R)
    end
    # only the strict upper triangle is used, R having a unit diagonal
    R .= tau .* R
end

# W = op(T) W (side 'L') or W = W op(T) (side 'R'), with op(T) = T or Tᴴ
function householder_t!(side, adj, R, tau, W)
    T = eltype(W)
    transr = T <: Real ? 'T' : 'C'
    if side == 'L'
        if adj
            cuBLAS.trsm!('L', 'U', transr, 'U', one(T), R, W)
            W .= conj.(tau) .* W
        else
            W .= tau .* W
            cuBLAS.trsm!('L', 'U', 'N', 'U', one(T), R, W)
        end
    else
        if adj
            W .= W .* adjoint(tau)
            cuBLAS.trsm!('R', 'U', transr, 'U', one(T), R, W)
        else
            cuBLAS.trsm!('R', 'U', 'N', 'U', one(T), R, W)
            W .= W .* transpose(tau)
        end
    end
    W
end

# apply H = I - V T Vᴴ, or Hᴴ, to C from the left or the right, as LAPACK's larfb does.
# V's leading block is read as unit lower triangular, so its upper triangle is not
# referenced. W is scratch space of size (jb, n) for side 'L' or (m, jb) for side 'R'.
function apply_block_reflector!(side, adj, V, R, tau, C, W)
    T = eltype(C)
    mv, jb = size(V)
    transv = T <: Real ? 'T' : 'C'
    V1 = view(V, 1:jb, :)
    V2 = view(V, (jb+1):mv, :)
    if side == 'L'
        C1 = view(C, 1:jb, :)
        C2 = view(C, (jb+1):mv, :)
        cuBLAS.trmm!('L', 'L', transv, 'U', one(T), V1, C1, W)      # W = Vᴴ C
        mv > jb && cuBLAS.gemm!(transv, 'N', one(T), V2, C2, one(T), W)
        householder_t!('L', adj, R, tau, W)                         # W = op(T) W
        mv > jb && cuBLAS.gemm!('N', 'N', -one(T), V2, W, one(T), C2)
        cuBLAS.trmm!('L', 'L', 'N', 'U', one(T), V1, W, W)          # C -= V W
        C1 .-= W
    else
        C1 = view(C, :, 1:jb)
        C2 = view(C, :, (jb+1):mv)
        cuBLAS.trmm!('R', 'L', 'N', 'U', one(T), V1, C1, W)         # W = C V
        mv > jb && cuBLAS.gemm!('N', 'N', one(T), C2, V2, one(T), W)
        householder_t!('R', adj, R, tau, W)                         # W = W op(T)
        mv > jb && cuBLAS.gemm!('N', transv, -one(T), W, V2, one(T), C2)
        cuBLAS.trmm!('R', 'L', transv, 'U', one(T), V1, W, W)       # C -= W Vᴴ
        C1 .-= W
    end
    C
end

function ormqr_blocked!(side::Char, trans::Char, A::StridedCuMatrix{T},
                        tau::StridedCuVector{T}, C::StridedCuVecOrMat{T},
                        nb::Integer) where {T}
    mA = size(A, 1)
    k = length(tau)
    k == 0 && return C
    if C isa AbstractVector && !(C isa CuVector)
        # cuBLAS needs a matrix, which a reshaped view is not
        c = copy(C)
        ormqr_blocked!(side, trans, A, tau, c, nb)
        copyto!(C, c)
        unsafe_free!(c)
        return C
    end
    Cm = C isa AbstractVector ? reshape(C, length(C), 1) : C
    nb = min(nb, k)
    R = similar(A, nb, nb)
    V1 = similar(A, nb, nb)
    W = side == 'L' ? similar(A, nb, size(Cm, 2)) : similar(A, size(Cm, 1), nb)
    try
        # Q = H(1) ⋯ H(k), so Qᴴ C and C Q apply the panels front to back, Q C and C Qᴴ
        # back to front. The reflectors of panel j only touch rows (or columns) j+1:mA.
        forward = (side == 'L') != (trans == 'N')
        for j in (forward ? (0:nb:k-1) : reverse(0:nb:k-1))
            jb = min(nb, k - j)
            V = view(A, (j+1):mA, (j+1):(j+jb))
            tauj = view(tau, (j+1):(j+jb))
            Rj = view(R, 1:jb, 1:jb)
            householder_ut!(Rj, view(V1, 1:jb, 1:jb), V, tauj)
            Cj = side == 'L' ? view(Cm, (j+1):mA, :) : view(Cm, :, (j+1):mA)
            Wj = side == 'L' ? view(W, 1:jb, :) : view(W, :, 1:jb)
            apply_block_reflector!(side, trans != 'N', V, Rj, tauj, Cj, Wj)
        end
    finally
        unsafe_free!(R)
        unsafe_free!(V1)
        unsafe_free!(W)
    end
    C
end

# overwrite A[:, 1:n] with the first n columns of Q, back to front as LAPACK's orgqr does
function orgqr_blocked!(A::StridedCuMatrix{T}, tau::StridedCuVector{T}, n::Integer,
                        nb::Integer) where {T}
    m = size(A, 1)
    k = length(tau)
    if k < n
        view(A, :, (k+1):n) .= T.((1:m) .== ((k+1):n)')
    end
    k == 0 && return A
    nb = min(nb, k)
    R = similar(A, nb, nb)
    V1 = similar(A, nb, nb)
    Y = similar(A, nb, nb)
    W = similar(A, nb, n)
    try
        for j in reverse(0:nb:k-1)
            jb = min(nb, k - j)
            V = view(A, (j+1):m, (j+1):(j+jb))
            tauj = view(tau, (j+1):(j+jb))
            Rj = view(R, 1:jb, 1:jb)
            V1j = view(V1, 1:jb, 1:jb)
            householder_ut!(Rj, V1j, V, tauj)

            # the columns to the right already hold H(j+jb+1) ⋯ H(k) applied to I
            if j + jb < n
                apply_block_reflector!('L', false, V, Rj, tauj, view(A, (j+1):m, (j+jb+1):n),
                                       view(W, 1:jb, 1:(n-j-jb)))
            end

            # this panel's columns of I - V T Vᴴ are [I; 0] - V Y with Y = T V1ᴴ, which is
            # upper triangular, so they can be formed in place over V
            Yj = view(Y, 1:jb, 1:jb)
            Yj .= adjoint(V1j)
            householder_t!('L', false, Rj, tauj, Yj)
            copyto!(view(V, 1:jb, :), V1j)
            cuBLAS.trmm!('R', 'U', 'N', 'N', -one(T), Yj, V, V)
            view(V, 1:jb, :) .+= T.((1:jb) .== (1:jb)')
            fill!(view(A, 1:j, (j+1):(j+jb)), zero(T))
        end
    finally
        unsafe_free!(R)
        unsafe_free!(V1)
        unsafe_free!(Y)
        unsafe_free!(W)
    end
    A
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
                        C::StridedCuVecOrMat{$elty};
                        blocksize::Union{Nothing,Integer}=nothing)

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

            lwork = Ref{Cint}(0)
            if blocksize === nothing
                if max(m, n, lda, ldc) > typemax(Cint) || prefer_blocked(mA, k)
                    blocksize = HOUSEHOLDER_BLOCKSIZE
                else
                    status = $(Symbol(:unchecked_, bname))(dh, side, trans, m, n, k, A, lda,
                                                           tau, C, ldc, lwork)
                    if status == CUSOLVER_STATUS_INVALID_VALUE || lwork[] < 0
                        blocksize = HOUSEHOLDER_BLOCKSIZE
                    elseif status != CUSOLVER_STATUS_SUCCESS
                        throw_api_error(status)
                    end
                end
            end
            if blocksize !== nothing
                return ormqr_blocked!(side, trans, A, tau, C, blocksize)
            end

            with_workspace(dh.workspace_gpu, lwork[] * sizeof($elty)) do buffer
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
        function orgqr!(A::StridedCuMatrix{$elty}, tau::CuVector{$elty};
                        blocksize::Union{Nothing,Integer}=nothing)
            m = size(A , 1)
            n = min(m, size(A, 2))
            lda = max(1, stride(A, 2))
            k = length(tau)
            dh = dense_handle()

            # this workspace query does not fail on overflow, but wraps around: to a
            # negative size at first, and past 2^32 elements to a small positive one that
            # makes cuSOLVER write out of bounds. The workspace is no larger than Q, up to
            # lower-order terms, so bounding `lda * n` rules out the latter.
            lwork = Ref{Cint}(0)
            if blocksize === nothing
                if lda * n > typemax(Cint) || prefer_blocked(m, k)
                    blocksize = HOUSEHOLDER_BLOCKSIZE
                else
                    status = $(Symbol(:unchecked_, bname))(dh, m, n, k, A, lda, tau, lwork)
                    if status == CUSOLVER_STATUS_INVALID_VALUE || lwork[] < 0
                        blocksize = HOUSEHOLDER_BLOCKSIZE
                    elseif status != CUSOLVER_STATUS_SUCCESS
                        throw_api_error(status)
                    end
                end
            end

            if blocksize !== nothing
                orgqr_blocked!(A, tau, n, blocksize)
            else
                with_workspace(dh.workspace_gpu, lwork[] * sizeof($elty)) do buffer
                    $fname(dh, m, n, k, A, lda, tau,
                           buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
                end

                info = @allowscalar dh.info[1]
                chkargsok(BlasInt(info))
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
            dh = dense_handle()

            # this routine reports its workspace size as a 32-bit element count, and
            # its query rejects matrices of about 2^31 elements and more (and fewer
            # when they are very tall). The same algorithm in the 64-bit API has no
            # such limit and the same interface, so hand those problems to it.
            max(m, n, lda) > typemax(Cint) && return Xgesvd!(jobu, jobvt, A)
            lwork = Ref{Cint}(0)
            status = $(Symbol(:unchecked_, bname))(dh, m, n, lwork)
            if status == CUSOLVER_STATUS_INVALID_VALUE || lwork[] < 0
                return Xgesvd!(jobu, jobvt, A)
            elseif status != CUSOLVER_STATUS_SUCCESS
                throw_api_error(status)
            end

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
        # whether the 32-bit API can serve this problem at all. The workspace query
        # reports a 32-bit element count: it fails for matrices of about 2^31
        # elements and more (fewer when very tall or wide), and for workspaces in
        # [2^31, 2^32) elements -- square matrices from about 32767^2 -- it reports
        # success with a size of 0. Ask with placeholder outputs so that nothing of
        # the size of A needs to be allocated to find out.
        function gesvdj_fits(jobz::Char, econ::Int, A::StridedCuMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            max(m, n, lda) > typemax(Cint) && return false
            dh = dense_handle()
            X = CuVector{$elty}(undef, 1)
            S = CuVector{$relty}(undef, 1)
            params = Ref{gesvdjInfo_t}(C_NULL)
            cusolverDnCreateGesvdjInfo(params)
            lwork = Ref{Cint}(0)
            try
                status = $(Symbol(:unchecked_, bname))(dh, jobz, econ, m, n, A, lda, S,
                                                       X, max(1, m), X, max(1, n), lwork,
                                                       params[])
                status == CUSOLVER_STATUS_INVALID_VALUE && return false
                status == CUSOLVER_STATUS_SUCCESS || throw_api_error(status)
                lwork[] > 0
            finally
                cusolverDnDestroyGesvdjInfo(params[])
                unsafe_free!(X)
                unsafe_free!(S)
            end
        end

        function gesvdj!(jobz::Char,
                         econ::Int,
                         A::StridedCuMatrix{$elty};
                         tol::$relty=eps($relty),
                         max_sweeps::Int=100)
            m,n     = size(A)
            lda     = max(1, stride(A, 2))
            gesvdj_fits(jobz, econ, A) ||
                throw(ArgumentError("CUSOLVER's gesvdj cannot handle a $m×$n matrix: its 32-bit workspace size overflows. Use svd(A; alg=QRAlgorithm()) or Xgesvd! instead."))
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
