# The wrappers in this file use cuSOLVER's 64-bit generic API (cusolverDnX*), which is
# available from cuSOLVER 11. Where the legacy 32-bit API offers the same routine, it is
# used on older versions.

@public potrf!, potrs!, getrf!, getrs!, geqrf!, sytrs!, trtri!, larft!, gesvd!, gesvdp!,
        gesvdr!, syevd!, syevdx!, geev!, syevBatched!

"""
    potrf!(uplo, A) -> A, info

Compute the Cholesky factorization of the Hermitian positive definite matrix `A` in place,
in the triangle selected by `uplo` (`'U'` or `'L'`). `info > 0` means that the leading minor
of that order is not positive definite.

Wraps `cusolverDnXpotrf`, or `cusolverDn<t>potrf` before cuSOLVER 11.
"""
function potrf!(uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_potrf!(uplo, A)
    chkuplo(uplo)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXpotrf_bufferSize(dh, params, uplo, n,
                                    T, A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXpotrf(dh, params, uplo, n, T, A, lda, T,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, flag
end

"""
    potrs!(uplo, A, B) -> B

Solve `A X = B` in place in `B`, using the Cholesky factorization of `A` computed by
[`potrf!`](@ref) with the same `uplo`.

Wraps `cusolverDnXpotrs`, or `cusolverDn<t>potrs` before cuSOLVER 11.
"""
function potrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_potrs!(uplo, A, B)
    chkuplo(uplo)
    n = checksquare(A)
    p, nrhs = size(B, 1), size(B, 2)
    (p ≠ n) && throw(DimensionMismatch("first dimension of B, $p, must match second dimension of A, $n"))
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    cusolverDnXpotrs(dh, params, uplo, n, nrhs, T, A, lda, T, B, ldb, dh.info)

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

"""
    getrf!(A[, ipiv]) -> A, ipiv, info

Compute the LU factorization of `A` with partial pivoting in place, storing the pivot
indices in `ipiv`, a `CuVector{Int64}` or `CuVector{Cint}` of length `min(size(A)...)`. When
`ipiv` is not given, a `CuVector{Cint}` is allocated. `info > 0` means that `U[info, info]`
is exactly zero.

Wraps `cusolverDnXgetrf` for 64-bit pivots, and `cusolverDn<t>getrf` for 32-bit pivots or
before cuSOLVER 11.
"""
function getrf!(A::StridedCuMatrix{T}, ipiv::CuVector{Int64}) where {T <: BlasFloat}
    if version() < v"11"
        ipiv32 = CuVector{Cint}(undef, length(ipiv))
        _, _, info = getrf!(A, ipiv32)
        ipiv .= ipiv32
        return A, ipiv, info
    end
    m, n = size(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgetrf_bufferSize(dh, params, m, n, T,
                                    A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgetrf(dh, params, m, n, T, A, lda, ipiv,
                         T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, ipiv, flag
end

"""
    getrs!(trans, A, ipiv, B) -> B

Solve `op(A) X = B` in place in `B`, where `trans` selects `op` (`'N'`, `'T'` or `'C'`),
using the LU factorization `A` and pivots `ipiv` computed by [`getrf!`](@ref).

Wraps `cusolverDnXgetrs` for 64-bit pivots, and `cusolverDn<t>getrs` for 32-bit pivots or
before cuSOLVER 11.
"""
function getrs!(trans::Char, A::StridedCuMatrix{T}, ipiv::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11" && return getrs!(trans, A, CuVector{Cint}(ipiv), B)
    # support trans = 'C' for real matrices
    trans = T <: Real && trans == 'C' ? 'T' : trans
    chktrans(trans)
    n = checksquare(A)
    if size(B, 1) != n
        throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match dimension of A, $n"))
    end
    if length(ipiv) != n
        throw(DimensionMismatch("length of ipiv, $(length(ipiv)), must match dimension of A, $n"))
    end
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    cusolverDnXgetrs(dh, params, trans, n, nrhs, T, A, lda, ipiv, T, B, ldb, dh.info)

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

"""
    geqrf!(A[, tau]) -> A, tau

Compute the QR factorization of `A` in place: `R` in the upper triangle, and `Q` as
Householder reflectors below it with their scalar factors in `tau`. Use [`orgqr!`](@ref) to
form `Q` and [`ormqr!`](@ref) to multiply by it.

Wraps `cusolverDnXgeqrf`, or `cusolverDn<t>geqrf` before cuSOLVER 11.
"""
function geqrf!(A::StridedCuMatrix{T}, tau::CuVector{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_geqrf!(A, tau)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeqrf_bufferSize(dh, params, m, n, T, A,
                                    lda, T, tau, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeqrf(dh, params, m, n, T, A,
                         lda, T, tau, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, tau
end

function geqrf!(A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    tau = CuVector{T}(undef, min(m,n))
    geqrf!(A, tau)
end

"""
    sytrs!(uplo, A, [ipiv,] B) -> B

Solve `A X = B` in place in `B`, using the factorization of the symmetric matrix `A`
computed by [`sytrf!`](@ref) with the same `uplo`. `ipiv` is a `CuVector{Int64}`: convert
the pivots `sytrf!` returns with `CuVector{Int64}(ipiv)`. Leave it out for a factorization
computed with `pivoting=false`.

Wraps `cusolverDnXsytrs`. Requires cuSOLVER 11.2 (CUDA 11.4).
"""
function sytrs!(uplo::Char, A::StridedCuMatrix{T}, p::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11.2" && error("This operation requires cuSOLVER 11.2 or later.")
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dh, uplo, n, nrhs, T, A,
                                    lda, p, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu,
                    bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dh, uplo, n, nrhs, T, A, lda, p,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

function sytrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11.2" && error("This operation requires cuSOLVER 11.2 or later.")
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dh, uplo, n, nrhs, T, A,
                                    lda, CU_NULL, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu,
                    bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dh, uplo, n, nrhs, T, A, lda, CU_NULL,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

"""
    trtri!(uplo, diag, A) -> A

Invert the triangular matrix `A` in place. `uplo` selects the triangle, and `diag` is `'U'`
for a unit diagonal, which is not read, or `'N'`.

Wraps `cusolverDnXtrtri`. Requires cuSOLVER 11.2 (CUDA 11.4).
"""
function trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11.2" && error("This operation requires cuSOLVER 11.2 or later.")
    chkuplo(uplo)
    chkdiag(diag)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXtrtri_bufferSize(dh, uplo, diag, n, T, A, lda, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXtrtri(dh, uplo, diag, n, T, A, lda,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu),
                         dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A
end

"""
    larft!(direct, storev, V, tau, T) -> T

Form the triangular factor `T` of the block reflector `H = I - V T Vᴴ` of the `k`
elementary reflectors stored in the columns of the `n×k` matrix `V`, with scalar factors
`tau`, as LAPACK's `larft` does. `direct` is `'F'` for `H = H(1)⋯H(k)` and `'B'` for
`H = H(k)⋯H(1)`; `storev` must be `'C'`.

Wraps `cusolverDnXlarft`. Requires cuSOLVER 11.6.
"""
function larft!(direct::Char, storev::Char, v::StridedCuMatrix{T}, tau::StridedCuVector{T}, t::StridedCuMatrix{T}) where {T <: BlasFloat}
    cuSOLVER.version() < v"11.6.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    n, k = size(v)
    ktau = length(tau)
    mt, nt = size(t)
    (storev != 'C') && throw(ArgumentError("Only storev = 'C' is supported."))
    (n < k) && throw(ArgumentError("The number of elementary reflectors ($k) must be lower or equal to the order of block reflector H ($n)."))
    (ktau != k) && throw(ArgumentError("The length of tau ($ktau) is not equal to the number of elementary reflectors ($k)."))
    (mt != k || nt != k) && throw(ArgumentError("The size of the triangular factor of the block reflector is ($mt, $nt) and must be ($k, $k)."))
    ldv = max(1, stride(v, 2))
    ldt = max(1, stride(t, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXlarft_bufferSize(dh, params, direct, storev, n, k, T,
                                    v, ldv, T, tau, T, t, ldt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXlarft(dh, params, direct, storev, n, k, T, v, ldv, T, tau, T, t,
                         ldt, T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu))
    end

    t
end

"""
    gesvd!(jobu, jobvt, A) -> U, S, Vt

Compute the singular value decomposition `A = U * Diagonal(S) * Vt` of an `m×n` matrix with
`m ≥ n`, using the QR algorithm. `jobu` and `jobvt` select which columns of `U` and rows of
`Vt` to compute: `'A'` for all, `'S'` for the first `min(m, n)`, `'O'` to overwrite `A` with
them, and `'N'` for none. For `'O'` and `'N'`, `CU_NULL` is returned in their place. `A` is
destroyed. Throws a `LAPACKException` if the algorithm does not converge.

Wraps `cusolverDnXgesvd`, or `cusolverDn<t>gesvd` before cuSOLVER 11.
"""
function gesvd!(jobu::Char, jobvt::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_gesvd!(jobu, jobvt, A)
    m, n = size(A)
    R = real(T)
    (m < n) && throw(ArgumentError("The number of rows of A ($m) must be greater or equal to the number of columns of A ($n)"))
    k = min(m, n)
    U = if jobu == 'A'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'S'
        CuMatrix{T}(undef, m, k)
    elseif jobu == 'N' || jobu == 'O'
        CU_NULL
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    Σ = CuVector{R}(undef, k)
    Vt = if jobvt == 'A'
        CuMatrix{T}(undef, n, n)
    elseif jobvt == 'S'
        CuMatrix{T}(undef, k, n)
    elseif jobvt == 'N' || jobvt == 'O'
        CU_NULL
    else
        throw(ArgumentError("jobvt is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldvt = Vt == CU_NULL ? 1 : max(1, stride(Vt, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvd_bufferSize(dh, params, jobu, jobvt,
                                    m, n, T, A, lda, R, Σ, T, U, ldu,
                                    T, Vt, ldvt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvd(dh, params, jobu, jobvt, m, n, T, A,
                         lda, R, Σ, T, U, ldu, T, Vt, ldvt, T, buffer_gpu,
                         sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    U, Σ, Vt
end

"""
    gesvdp!(jobz, econ, A) -> U, S, V, err     # jobz = 'V'
    gesvdp!(jobz, econ, A) -> S, err           # jobz = 'N'

Compute the singular value decomposition `A = U * Diagonal(S) * V'` using the polar
decomposition. With `econ = true`, `U` and `V` have `min(size(A)...)` columns. `err` is an
estimate of the error in the singular values. `A` is destroyed.

Wraps `cusolverDnXgesvdp`. Requires cuSOLVER 11.
"""
function gesvdp!(jobz::Char, econ::Bool, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    p = min(m, n)
    R = real(T)
    jobz in ('N', 'V') || throw(ArgumentError("jobz is incorrect. The values accepted are 'V' and 'N'."))

    U = CuMatrix{T}(undef, m, econ ? p : m)
    V = CuMatrix{T}(undef, n, econ ? p : n)
    Σ = CuVector{R}(undef, p)
    lda = max(1, stride(A, 2))
    ldu = max(1, stride(U, 2))
    ldv = max(1, stride(V, 2))
    h_err_sigma = Ref{Cdouble}(0)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdp_bufferSize(dh, params, jobz, econ, m,
                                     n, T, A, lda, R, Σ, T, U, ldu, T, V,
                                     ldv, T, out_gpu, out_cpu)

        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdp(dh, params, jobz, econ, m, n, T, A, lda, R,
                          Σ, T, U, ldu, T, V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), dh.info, h_err_sigma)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    if jobz == 'N'
        unsafe_free!(U)
        unsafe_free!(V)
        return Σ, h_err_sigma[]
    elseif jobz == 'V'
        return U, Σ, V, h_err_sigma[]
    end
end

"""
    gesvdr!(jobu, jobv, A, k; niters=2, p=2k) -> U, S, V

Compute a rank-`k` approximation `A ≈ U[:, 1:k] * Diagonal(S[1:k]) * V[:, 1:k]'` of `A`
with a randomized algorithm that uses `p` oversampling vectors and `niters` power
iterations. The singular vectors in `U` and `V` are only computed when `jobu` and `jobv` are
`'S'`, not when they are `'N'`. `A` is destroyed.

Wraps `cusolverDnXgesvdr`. Requires cuSOLVER 11.1 (CUDA 11.2).
"""
function gesvdr!(jobu::Char, jobv::Char, A::StridedCuMatrix{T}, k::Integer;
                  niters::Integer=2, p::Integer=2*k) where {T <: BlasFloat}
    version() < v"11.1" && error("This operation requires cuSOLVER 11.1 or later.")
    m, n = size(A)
    ℓ = min(m,n)
    p = min(p, ℓ-k)  # Ensure that p + k ≤ ℓ
    (1 ≤ k ≤ ℓ) || throw(ArgumentError("illegal choice of parameter k = $k, which must be between 1 and min(m,n) = $ℓ"))
    (k+p ≤ ℓ) || throw(ArgumentError("illegal choice of parameters k = $k and p = $p, which must satisfy k+p ≤ min(m,n) = $ℓ"))
    R = real(T)
    U = if jobu == 'S'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'N'
        CuMatrix{T}(undef, m, ℓ)
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'S' and 'N'."))
    end
    Σ = CuVector{R}(undef, ℓ)
    V = if jobv == 'S'
        CuMatrix{T}(undef, n, n)
    elseif jobv == 'N'
        CuMatrix{T}(undef, n, ℓ)
    else
        throw(ArgumentError("jobv is incorrect. The values accepted are 'S' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldv = V == CU_NULL ? 1 : max(1, stride(V, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdr_bufferSize(dh, params, jobu, jobv,
                                     m, n, k, p, niters, T, A, lda, R, Σ, T,
                                     U, ldu, T, V, ldv, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdr(dh, params, jobu, jobv, m, n,
                          k, p, niters, T, A, lda, R, Σ, T, U, ldu, T,
                          V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    U, Σ, V
end

"""
    syevd!(jobz, uplo, A) -> W         # jobz = 'N'
    syevd!(jobz, uplo, A) -> W, A      # jobz = 'V'

Compute the eigenvalues `W`, in ascending order, of the symmetric or Hermitian matrix `A`
from the triangle selected by `uplo`, using a divide-and-conquer algorithm. With
`jobz = 'V'`, the eigenvectors overwrite `A`.

Wraps `cusolverDnXsyevd`, or `cusolverDn<t>syevd`/`cusolverDn<t>heevd` before cuSOLVER 11.
"""
function syevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_syevd!(jobz, uplo, A)
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevd_bufferSize(dh, params, jobz, uplo, n,
                                    T, A, lda, R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevd(dh, params, jobz, uplo, n, T, A,
                         lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

"""
    syevdx!(jobz, range, uplo, A; vl=0.0, vu=Inf, il=1, iu=n) -> W, m        # jobz = 'N'
    syevdx!(jobz, range, uplo, A; vl=0.0, vu=Inf, il=1, iu=n) -> W, A, m     # jobz = 'V'

Like [`syevd!`](@ref), but compute only some of the eigenvalues: all of them with
`range = 'A'`, those in the interval `(vl, vu]` with `range = 'V'`, and the `il`-th through
`iu`-th with `range = 'I'`. The first `m` entries of `W`, and columns of `A`, hold the
result.

Wraps `cusolverDnXsyevdx`. Requires cuSOLVER 11.
"""
function syevdx!(jobz::Char, range::Char, uplo::Char, A::StridedCuMatrix{T};
                  vl::Real=0.0, vu::Real=Inf, il::Integer=1, iu::Integer=0) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    iu = iu == 0 ? n : iu
    (range == 'I') && !(1 ≤ il ≤ iu ≤ n) && throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
    (range == 'V') && (vl ≥ vu) && throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n)
    vl = Ref{R}(vl)
    vu = Ref{R}(vu)
    h_meig = Ref{Int64}(0)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevdx_bufferSize(dh, params, jobz, range, uplo, n,
                                     T, A, lda, vl, vu, il, iu, h_meig,
                                     R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevdx(dh, params, jobz, range, uplo, n, T, A,
                          lda, vl, vu, il, iu, h_meig, R, W, T, buffer_gpu,
                          sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    if jobz == 'N'
        return W, h_meig[]
    elseif jobz == 'V'
        return W, A, h_meig[]
    end
end

"""
    geev!(jobvl, jobvr, A) -> W, VL, VR

Compute the eigenvalues `W` of the square matrix `A`, which are complex also for real `A`,
and with `jobvl` and `jobvr` set to `'V'` its left and right eigenvectors (`'N'` returns
`CU_NULL` in their place). For real `A`, the eigenvectors are real and stored as LAPACK's
`geev` does: a complex conjugate pair occupies two columns, holding the real and imaginary
part of the first vector. `A` is destroyed.

Wraps `cusolverDnXgeev`. Requires cuSOLVER 11.7.1 (CUDA 12.6.2).
"""
function geev!(jobvl::Char, jobvr::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    cuSOLVER.version() < v"11.7.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    n = checksquare(A)
    VL = if jobvl == 'V'
        CuMatrix{T}(undef, n, n)
    elseif jobvl == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvl is incorrect. The values accepted are 'V' and 'N'."))
    end
    C = T <: Real ? Complex{T} : T
    W = CuVector{C}(undef, n)
    VR = if jobvr == 'V'
        CuMatrix{T}(undef, n, n)
    elseif jobvr == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvr is incorrect. The values accepted are 'V' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldvl = VL == CU_NULL ? 1 : max(1, stride(VL, 2))
    ldvr = VR == CU_NULL ? 1 : max(1, stride(VR, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeev_bufferSize(dh, params, jobvl, jobvr, n, T, A,
                                   lda, C, W, T, VL, ldvl, T, VR, ldvr,
                                   T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeev(dh, params, jobvl, jobvr, n, T, A, lda, C,
                        W, T, VL, ldvl, T, VR, ldvr, T, buffer_gpu,
                        sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    return W, VL, VR
end

"""
    syevBatched!(jobz, uplo, A) -> W         # jobz = 'N'
    syevBatched!(jobz, uplo, A) -> W, A      # jobz = 'V'

Compute the eigenvalues, and with `jobz = 'V'` the eigenvectors, of each symmetric or
Hermitian matrix `A[:, :, i]` of a 3-D array, as [`syevd!`](@ref) does. Column `i` of `W`
holds the eigenvalues of matrix `i`.

Wraps `cusolverDnXsyevBatched`. Requires cuSOLVER 11.7.1.
"""
function syevBatched!(jobz::Char, uplo::Char, A::StridedCuArray{T, 3}) where {T <: BlasFloat}
    minimum_version = v"11.7.1"
    cuSOLVER.version() < minimum_version && throw(ErrorException("This operation requires cuSOLVER
        $(minimum_version) or later. Current cuSOLVER version: $(cuSOLVER.version())."))
    chkuplo(uplo)
    n = checksquare(A)
    batch_size = size(A, 3)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuMatrix{R}(undef, n, batch_size)
    params = CuSolverParameters()
    dh = dense_handle()
    resize!(dh.info, batch_size)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevBatched_bufferSize(
            dh, params, jobz, uplo, n,
            T, A, lda, R, W, T, out_gpu, out_cpu, batch_size
        )
        return out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevBatched(
            dh, params, jobz, uplo, n, T, A,
            lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
            buffer_cpu, sizeof(buffer_cpu), dh.info, batch_size
        )
    end

    info = @allowscalar collect(dh.info)
    for i in 1:batch_size
        chkargsok(info[i] |> BlasInt)
    end

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

# LAPACK
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        LinearAlgebra.LAPACK.sytrs!(uplo::Char, A::StridedCuMatrix{$elty}, p::CuVector{Int64}, B::StridedCuVecOrMat{$elty}) = cuSOLVER.sytrs!(uplo, A, p, B)
        LinearAlgebra.LAPACK.trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{$elty}) = cuSOLVER.trtri!(uplo, diag, A)
    end
end
